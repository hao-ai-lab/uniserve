//! Engine-loop construction, event parking, and owner-thread execution.

use super::*;
use uniserve_worker_ipc::{ForwardMode, MediaCall};

/// Intersects model requirements with capacities actually loaded by the worker.
fn resolve_generation_limits(
    mut limits: uniserve_core::GenerationLimits,
    info: &WorkerInfo,
) -> uniserve_core::GenerationLimits {
    let supports = |kind| info.supported_calls.contains(&kind);
    let mut available = uniserve_core::GenerationFeatures::empty();
    if supports(CallKind::Forward(ForwardMode::Prefill))
        && supports(CallKind::Forward(ForwardMode::Decode))
    {
        available.insert(uniserve_core::GenerationFeatures::UNDERSTANDING);
    }
    if supports(CallKind::Media(MediaCall::VisionEncoding)) {
        available.insert(uniserve_core::GenerationFeatures::VISION_ENCODE);
    }
    if supports(CallKind::Media(MediaCall::LatentEncoding)) {
        available.insert(uniserve_core::GenerationFeatures::LATENT_ENCODE);
    }
    if supports(CallKind::Media(MediaCall::LatentPreparation))
        && supports(CallKind::Media(MediaCall::Denoising))
        && supports(CallKind::Media(MediaCall::ImageDecoding))
    {
        available.insert(uniserve_core::GenerationFeatures::IMAGE_GENERATION);
    }
    limits.features &= available;
    limits.max_latent_units = limits.max_latent_units.min(info.latent_capacity_units());
    let latent_bound = limits.max_latent_units.min(u64::from(u32::MAX)) as u32;
    limits.max_vae_grid_tokens = limits.max_vae_grid_tokens.min(latent_bound);
    limits.max_vit_grid_tokens = limits.max_vit_grid_tokens.min(info.max_batch_tokens);
    let feature_bytes = info.encoder_entry_bytes.min(info.buffer_pool_bytes);
    limits.max_latent_feature_bytes = limits.max_latent_feature_bytes.min(feature_bytes);
    limits.max_vision_feature_bytes = limits.max_vision_feature_bytes.min(feature_bytes);
    limits.encoder_cache_entries = limits.encoder_cache_entries.min(info.encoder_cache_entries);
    if info.buffer_pool_bytes == 0 {
        limits.encoder_cache_entries = 0;
    }
    limits
}

impl Scheduler {
    /// Constructs an engine loop with the default scheduler configuration.
    pub fn new(executor: Box<dyn Executor>, ctrl: SpecialTokenIds, max_batch: usize) -> Self {
        Self::with_config(
            executor,
            ctrl,
            SchedulerConfig {
                max_batch,
                ..Default::default()
            },
        )
    }

    /// Constructs an engine loop with an explicit scheduling policy.
    pub fn with_policy(
        executor: Box<dyn Executor>,
        ctrl: SpecialTokenIds,
        max_batch: usize,
        policy: SchedulingPolicy,
    ) -> Self {
        Self::with_config(
            executor,
            ctrl,
            SchedulerConfig {
                max_batch,
                policy,
                ..Default::default()
            },
        )
    }

    /// Constructs an engine loop and infers its runtime family from worker capabilities.
    ///
    /// # Panics
    ///
    /// Panics when the executor exposes an invalid aggregate capacity view.
    pub fn with_config(
        executor: Box<dyn Executor>,
        ctrl: SpecialTokenIds,
        config: SchedulerConfig,
    ) -> Self {
        let info = executor
            .info()
            .runtime_info()
            .expect("executor exposes a valid runtime capacity view");

        // Capability families are mutually ordered from diffusion-only through
        // unified multimodal support to autoregressive-only execution.
        let calls = &info.supported_calls;
        let family = if calls.contains(&CallKind::Media(MediaCall::LatentPreparation))
            && !calls.contains(&CallKind::Forward(ForwardMode::Decode))
        {
            RuntimeFamily::Diffusion
        } else if calls.contains(&CallKind::Media(MediaCall::Denoising))
            || (calls.contains(&CallKind::Media(MediaCall::VisionEncoding))
                || calls.contains(&CallKind::Media(MediaCall::LatentEncoding)))
        {
            RuntimeFamily::Umm
        } else {
            RuntimeFamily::Ar
        };

        Self::with_config_for_family(executor, ctrl, config, family)
    }

    /// Constructs an engine loop for an explicit runtime family.
    pub fn with_config_for_family(
        executor: Box<dyn Executor>,
        ctrl: SpecialTokenIds,
        config: SchedulerConfig,
        family: RuntimeFamily,
    ) -> Self {
        let generation_limits = match family {
            RuntimeFamily::Umm => unbounded_umm_generation_limits(),
            RuntimeFamily::Ar | RuntimeFamily::Diffusion => uniserve_core::GenerationLimits {
                features: if family == RuntimeFamily::Ar {
                    uniserve_core::GenerationFeatures::UNDERSTANDING
                } else {
                    uniserve_core::GenerationFeatures::empty()
                },
                latent_downsample: 1,
                max_cfg_branches: 1,
                ..Default::default()
            },
        };
        Self::with_model_limits(
            executor,
            ctrl,
            config,
            family,
            uniserve_core::ModelDtype::BFloat16,
            generation_limits,
        )
    }

    /// Constructs an engine loop from explicit scheduler and model capabilities.
    ///
    /// Worker limits clamp scheduler concurrency and resource capacities.
    ///
    /// # Panics
    ///
    /// Panics when the executor exposes an invalid aggregate capacity view.
    pub fn with_model_limits(
        executor: Box<dyn Executor>,
        ctrl: SpecialTokenIds,
        mut config: SchedulerConfig,
        family: RuntimeFamily,
        model_dtype: uniserve_core::ModelDtype,
        generation_limits: uniserve_core::GenerationLimits,
    ) -> Self {
        let info = executor
            .info()
            .runtime_info()
            .expect("executor exposes a valid runtime capacity view");
        let generation_limits = resolve_generation_limits(generation_limits, &info);
        let latent_dtype = worker_float_dtype(Some(model_dtype));

        // Queue and batch limits cannot exceed the physical executor envelope.
        let max_batch_calls = info.max_batch_calls as usize;
        let max_batch_tokens = info.max_batch_tokens as usize;
        let transfer_capacity = (info.queue_depth as usize)
            .saturating_mul(max_batch_calls)
            .clamp(1, MAX_INFLIGHT_TRANSFERS);

        config.max_num_waiting = config.max_num_waiting.clamp(1, MAX_NUM_WAITING);

        // Unified runtimes reserve one physical slot for the image flow lineage.
        let flow_slot_reserve = usize::from(
            info.uses_kv()
                && info
                    .supported_calls
                    .contains(&CallKind::Media(MediaCall::Denoising)),
        );
        let request_pool_capacity = info.request_slots as usize;
        let main_request_capacity = request_pool_capacity
            .saturating_sub(flow_slot_reserve)
            .max(1);

        config.max_num_seqs = config
            .max_num_seqs
            .clamp(1, MAX_NUM_SEQS)
            .min(main_request_capacity);
        config.max_num_batched_tokens = config.max_num_batched_tokens.max(1).min(max_batch_tokens);
        if max_batch_calls > 0 {
            config.max_batch = config.max_batch.min(max_batch_calls.max(1));
        }

        // Memory and scheduler statistics share the resolved worker capacities.
        let cache = KVCacheManager::from_worker_info(&info);
        let stats = Arc::new(SchedulerStats::default());
        stats.kv_cache.num_blocks.store(
            cache.as_ref().map_or(0, |state| state.usable_blocks),
            Ordering::Relaxed,
        );

        // Environment switches select scheduling behavior without changing the
        // model capabilities.
        let denoise_step_burst = denoise_step_burst_from_env();

        // Runtime state remains single-owner; executors receive immutable batch
        // inputs assembled from these queues and request records.

        Self {
            executor,
            pending_submissions: VecDeque::new(),
            worker_affinity: HashMap::new(),
            cache,
            encoder_cache: crate::kv::EncoderCacheManager::new(
                generation_limits.encoder_cache_entries as usize,
            ),
            reserved_encoder_entries: 0,
            request_pool: RequestPool::new(info.request_slots as usize),
            latent_pool: LatentPool::new(info.latent_pages, info.latent_page_units),
            reserved_blocks: 0,
            buffer_pool: BufferPool::new(info.buffer_pool_bytes),
            encoder_buffers: HashMap::new(),
            info,
            generation_limits,
            family,
            ctrl,
            waiting: HashMap::new(),
            waiting_media: HashMap::new(),
            running: HashMap::new(),
            running_media: HashMap::new(),
            retiring_requests: HashMap::new(),
            transfer_capacity,
            num_pending_transfers: 0,
            batch_id: 0,
            next_arrival_seq: 1,
            pending_calls: HashMap::new(),
            pending_completions: HashMap::new(),
            pending_finishes: HashMap::new(),
            pending_batches: HashMap::new(),
            denoise_step_burst,
            latent_dtype,
            pending_commands: VecDeque::new(),
            pending_buffer_frees: HashMap::new(),
            engine_id: 1,
            next_product_generation: 1,
            next_request_epoch: 1,
            waiting_order: VecDeque::new(),
            waiting_media_order: VecDeque::new(),
            running_order: Vec::new(),
            output: output::OutputSender::default(),
            prefer_media: true,
            config,
            fatal: false,
            peak_calls_in_batch: 0,
            stats,
        }
    }

    /// Returns the active scheduling policy.
    pub fn policy(&self) -> SchedulingPolicy {
        self.config.policy
    }

    /// Returns the effective scheduler configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Enables or disables reusable prefix caching.
    pub fn set_prefix_cache(&mut self, on: bool) {
        if let Some(cache) = self.cache.as_mut() {
            cache.set_prefix_cache(on);
        }
    }

    /// Selects the hash algorithm used for prefix-cache keys.
    pub fn set_hash_algo(&mut self, algo: HashAlgo) {
        if let Some(cache) = self.cache.as_mut() {
            cache.set_hash_algo(algo);
        }
    }
    /// Configures the per-step token budget.
    pub fn set_token_budget(&mut self, tokens: usize) {
        self.config.max_num_batched_tokens = tokens.max(1);
    }

    /// Sets the token threshold above which prefill is chunked.
    pub fn set_long_prefill_threshold(&mut self, n: usize) {
        self.config.long_prefill_threshold = n.max(1);
    }

    /// Sets the resident sequence limit within the worker slot capacity.
    pub fn set_max_num_seqs(&mut self, n: usize) {
        let flow_slot_reserve = usize::from(
            self.info.uses_kv()
                && self
                    .info
                    .supported_calls
                    .contains(&CallKind::Media(MediaCall::Denoising)),
        );
        let capacity = self
            .request_pool
            .capacity()
            .saturating_sub(flow_slot_reserve)
            .max(1);
        self.config.max_num_seqs = n.clamp(1, capacity);
    }
    /// Caps waiting and terminal-output-retained request state.
    pub fn set_max_num_waiting(&mut self, n: usize) {
        self.config.max_num_waiting = n.clamp(1, MAX_NUM_WAITING);
    }

    /// Returns the executor's aggregate worker capabilities.
    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Returns the model limits supported by the loaded execution workers.
    pub fn generation_limits(&self) -> &uniserve_core::GenerationLimits {
        &self.generation_limits
    }

    /// Returns a shared handle to scheduler counters.
    pub fn stats_handle(&self) -> Arc<SchedulerStats> {
        self.stats.clone()
    }

    /// Returns shared access to a media request state.
    pub(super) fn media_state(&self, id: RequestId) -> Option<&MediaFlowState> {
        self.running_media.get(&id)
    }

    /// Returns mutable access to a media request state.
    pub(super) fn media_state_mut(&mut self, id: RequestId) -> Option<&mut MediaFlowState> {
        self.running_media.get_mut(&id)
    }

    /// Returns the active media request identifiers.
    pub(super) fn media_ids(&self) -> Vec<RequestId> {
        self.running_media.keys().copied().collect()
    }

    /// Takes the media state.
    pub(super) fn take_media_state(&mut self, id: RequestId) -> Option<MediaFlowState> {
        self.running_media.remove(&id)
    }

    /// Returns the number of running requests.
    pub(super) fn running_request_count(&self) -> usize {
        self.running.len().saturating_add(self.running_media.len())
    }

    /// Returns the number of pending requests.
    pub(super) fn pending_request_count(&self) -> usize {
        self.waiting_order
            .len()
            .saturating_add(self.waiting_media_order.len())
    }

    /// Runs the owner-thread control loop, blocking only when fully idle.
    ///
    /// Returns `true` if the engine died
    /// (executor/worker failure) rather than shutting down gracefully.
    pub fn run(mut self, rx: Receiver<Command>) -> bool {
        loop {
            // drain pending commands (non-blocking)
            let mut shutdown = false;
            loop {
                match rx.try_recv() {
                    Ok(cmd) => {
                        if self.handle_command(cmd) {
                            shutdown = true;
                            break;
                        }
                    }
                    Err(crossbeam_channel::TryRecvError::Empty) => break,
                    Err(crossbeam_channel::TryRecvError::Disconnected) => {
                        shutdown = true;
                        break;
                    }
                }
            }
            if shutdown {
                break;
            }

            let progressed = self.step_nonblocking();
            if self.fatal {
                tracing::error!("engine fatal: executor/worker died; stopping the control loop");
                self.abort_all_requests();
                let _ = self.executor.close();
                return true;
            }

            if !progressed {
                self.park_for_progress();
                if self.fatal {
                    tracing::error!(
                        "engine fatal: executor/worker died during park; stopping the control loop"
                    );
                    self.abort_all_requests();
                    let _ = self.executor.close();
                    return true;
                }
            }
        }
        // Graceful shutdown: report Aborted to everything still queued or
        // running before tearing the executor down (staged drain happened at
        // the HTTP layer; nothing in flight should just see a closed channel).
        self.abort_all_requests();
        let _ = self.executor.close();
        false
    }

    /// Parks until a result, command, worker death, CPU continuation, or output-capacity wake.
    ///
    /// The timeout is solely a liveness deadline.
    pub(super) fn park_for_progress(&mut self) {
        let _span = tracing::trace_span!("scheduler.park").entered();
        match self.executor.poll(IDLE_LIVENESS_POLL) {
            Ok(Some(result)) => self.apply_result(result),
            Ok(None) => {}
            Err(error) => self.on_executor_error(error),
        }
        self.stats
            .general
            .in_flight
            .store(self.pending_batches.len(), Ordering::Relaxed);
        self.publish_cache_stats();
    }
}
