use super::*;

impl Scheduler {
    pub fn new(executor: Box<dyn Executor>, ctrl: ControlTokens, max_batch: usize) -> Self {
        Self::with_config(
            executor,
            ctrl,
            SchedulerConfig {
                max_batch,
                ..Default::default()
            },
        )
    }

    pub fn with_policy(
        executor: Box<dyn Executor>,
        ctrl: ControlTokens,
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

    pub fn with_config(
        executor: Box<dyn Executor>,
        ctrl: ControlTokens,
        mut config: SchedulerConfig,
    ) -> Self {
        let caps = executor.caps().clone();
        let max_batch_ops = caps.max_batch_operations as usize;
        let max_batch_tokens = caps.max_batch_tokens as usize;
        let transfer_capacity = (caps.pipeline_depth as usize)
            .saturating_mul(max_batch_ops)
            .clamp(1, MAX_INFLIGHT_TRANSFERS);
        config.max_num_waiting = config.max_num_waiting.clamp(1, MAX_NUM_WAITING);
        let flow_slot_reserve =
            usize::from(caps.uses_kv() && caps.supported_work.contains(&ForwardMode::GenFlow));
        let request_pool_capacity = caps.max_request_pool_size as usize;
        let main_request_capacity = request_pool_capacity
            .saturating_sub(flow_slot_reserve)
            .max(1);
        config.max_num_seqs = config
            .max_num_seqs
            .clamp(1, MAX_NUM_SEQS)
            .min(main_request_capacity);
        config.max_num_batched_tokens = config.max_num_batched_tokens.max(1).min(max_batch_tokens);
        if max_batch_ops > 0 {
            config.max_batch = config.max_batch.min(max_batch_ops.max(1));
        }
        let kv = caps.uses_kv().then(|| {
            let block_pool = if caps.groups.is_empty() {
                BlockPool::new(caps.num_blocks as usize, caps.block_size as usize)
            } else {
                let mut offset = 0_u32;
                let specs: Vec<(uniserve_core::KvGroupKind, u32, u32)> = caps
                    .groups
                    .iter()
                    .map(|group| {
                        let spec = (group.kind, offset, group.num_blocks);
                        offset = offset.saturating_add(group.num_blocks);
                        spec
                    })
                    .collect();
                BlockPool::with_groups(caps.num_blocks as usize, caps.block_size as usize, &specs)
            };
            let usable_blocks = block_pool.request_page_capacity();
            KvSchedulerState {
                block_pool,
                coordinator: KvCacheCoordinator::default(),
                usable_blocks,
            }
        });
        let stats = Arc::new(SchedStats::default());
        stats.kv_cache.num_blocks.store(
            kv.as_ref().map_or(0, |state| state.usable_blocks),
            Ordering::Relaxed,
        );
        let caps_encoder_budget = caps.encoder_cache_budget as usize;
        let request_slots = RequestSlotPool::new(request_pool_capacity);
        let latent_pages = LatentPagePool::new(caps.num_latent_pages, caps.latent_page_units);
        let denoise_step_burst = denoise_step_burst_from_env();
        let flow_exclusive_batch = env::var(FLOW_EXCLUSIVE_BATCH_ENV)
            .is_ok_and(|raw| matches!(raw.trim(), "1" | "true" | "TRUE"));
        let mut trace_sink = crate::scheduler::bench_trace::SchedulerTraceSink::from_env();
        if let Some(sink) = trace_sink.as_mut() {
            sink.record(&json!({
                "event": "run_started",
                "at_s": now(),
                "pid": std::process::id(),
                "scheduler": {
                    "policy": config.policy,
                    "max_batch": config.max_batch,
                    "max_num_batched_tokens": config.max_num_batched_tokens,
                    "max_num_seqs": config.max_num_seqs,
                    "long_prefill_threshold": config.long_prefill_threshold,
                    "mixed_prefill_tokens": config.mixed_prefill_tokens,
                    "denoise_step_burst": denoise_step_burst,
                },
                "caps": {
                    "block_size": caps.block_size,
                    "num_blocks": caps.num_blocks,
                    "supported_work": &caps.supported_work,
                    "max_batch_operations": caps.max_batch_operations,
                    "max_batch_tokens": caps.max_batch_tokens,
                    "max_request_pool_size": caps.max_request_pool_size,
                    "pipeline_depth": caps.pipeline_depth,
                    "latent_page_units": caps.latent_page_units,
                    "num_latent_pages": caps.num_latent_pages,
                    "latent_width": caps.latent_width,
                    "latent_dtype": &caps.latent_dtype,
                    "latent_downsample": caps.latent_downsample,
                    "max_vae_grid_tokens": caps.max_vae_grid_tokens,
                    "max_vit_grid_tokens": caps.max_vit_grid_tokens,
                    "commit_marker_tokens": caps.commit_marker_tokens,
                    "gen_rope_advance": caps.gen_rope_advance,
                    "max_cfg_branches": caps.max_cfg_branches,
                    "resource_classes": &caps.resource_classes,
                },
            }));
        }
        let latent_dtype = worker_float_dtype(caps.latent_dtype);
        Self {
            executor,
            caps,
            kv,
            ctrl,
            pending: RequestQueue::new(config.policy),
            config,
            logits_pipeline: crate::scheduler::logits::default_pipeline(),
            enc_cache: EncoderCacheManager::new(caps_encoder_budget),
            reserved_encoder_entries: 0,
            request_slots,
            latent_pages,
            running: HashMap::new(),
            running_media: HashMap::new(),
            completed_outputs: HashMap::new(),
            retiring_sessions: HashMap::new(),
            order: Vec::new(),
            pending_media: VecDeque::new(),
            retiring_media: HashMap::new(),
            prefer_media: true,
            reserved_blocks: 0,
            transfer_capacity,
            inflight_transfers: 0,
            step_id: 0,
            inflight_ops: HashMap::new(),
            pending_completions: HashMap::new(),
            pending_finishes: HashMap::new(),
            denoise_step_burst,
            flow_exclusive_batch,
            fatal: false,
            latent_dtype,
            batch_started: HashMap::new(),
            prefill_steps: HashSet::new(),
            batch_partitions: HashMap::new(),
            batch_group_worker_exec_us: HashMap::new(),
            pending_controls: VecDeque::new(),
            control_batches: HashMap::new(),
            authority_id: 1,
            next_op_id: 1,
            next_completion_seq: 1,
            next_product_generation: 1,
            next_collective_seq: 1,
            next_epoch: 1,
            trace_sink,
            peak_ops_in_batch: 0,
            stats,
        }
    }

    pub fn policy(&self) -> SchedulingPolicy {
        self.config.policy
    }
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }
    pub fn set_prefix_cache(&mut self, on: bool) {
        if let Some(kv) = self.kv.as_mut() {
            kv.coordinator.set_prefix_enabled(on);
        }
    }
    pub fn set_hash_algo(&mut self, algo: HashAlgo) {
        if let Some(kv) = self.kv.as_mut() {
            kv.coordinator.set_hash_algo(algo);
        }
    }
    /// Configure the per-step token budget.
    pub fn set_token_budget(&mut self, tokens: usize) {
        self.config.max_num_batched_tokens = tokens.max(1);
    }
    pub fn set_long_prefill_threshold(&mut self, n: usize) {
        self.config.long_prefill_threshold = n.max(1);
    }
    pub fn set_max_num_seqs(&mut self, n: usize) {
        let flow_slot_reserve = usize::from(
            self.caps.uses_kv() && self.caps.supported_work.contains(&ForwardMode::GenFlow),
        );
        let capacity = self
            .request_slots
            .capacity()
            .saturating_sub(flow_slot_reserve)
            .max(1);
        self.config.max_num_seqs = n.clamp(1, capacity);
    }
    /// Cap waiting and terminal-output-retained request state.
    pub fn set_max_num_waiting(&mut self, n: usize) {
        self.config.max_num_waiting = n.clamp(1, MAX_NUM_WAITING);
    }
    pub fn caps(&self) -> &WorkerCapabilities {
        &self.caps
    }
    pub fn stats_handle(&self) -> Arc<SchedStats> {
        self.stats.clone()
    }

    pub(super) fn kv_state(&self) -> &KvSchedulerState {
        self.kv
            .as_ref()
            .expect("generation scheduling requires worker KV resources")
    }

    pub(super) fn free_kv_blocks(&self) -> usize {
        self.kv
            .as_ref()
            .map_or(0, |state| state.block_pool.free_request_pages())
    }

    pub(super) fn usable_kv_blocks(&self) -> usize {
        self.kv.as_ref().map_or(0, |state| state.usable_blocks)
    }

    pub(super) fn media_state(&self, id: RequestId) -> Option<&MediaFlowState> {
        self.running_media.get(&id)
    }

    pub(super) fn media_state_mut(&mut self, id: RequestId) -> Option<&mut MediaFlowState> {
        self.running_media.get_mut(&id)
    }

    pub(super) fn media_ids(&self) -> Vec<RequestId> {
        self.running_media.keys().copied().collect()
    }

    pub(super) fn take_media_state(&mut self, id: RequestId) -> Option<MediaFlowState> {
        self.running_media.remove(&id)
    }

    pub(super) fn running_request_count(&self) -> usize {
        self.running.len().saturating_add(self.running_media.len())
    }

    pub(super) fn pending_request_count(&self) -> usize {
        self.pending.len().saturating_add(self.pending_media.len())
    }

    /// The owner thread: block on the command channel when fully idle, else
    /// spin the schedule-ahead loop. Returns `true` if the engine died
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
                self.executor.shutdown();
                return true;
            }

            if !progressed {
                self.park_for_progress();
                if self.fatal {
                    tracing::error!(
                        "engine fatal: executor/worker died during park; stopping the control loop"
                    );
                    self.abort_all_requests();
                    self.executor.shutdown();
                    return true;
                }
            }
        }
        // Graceful shutdown: report Aborted to everything still queued or
        // running before tearing the executor down (staged drain happened at
        // the HTTP layer; nothing in flight should just see a closed channel).
        self.abort_all_requests();
        self.executor.shutdown();
        false
    }

    /// One park over result, command, worker-death, CPU-continuation, and
    /// output-capacity wakes. The timeout is solely a liveness deadline.
    pub(super) fn park_for_progress(&mut self) {
        let _span = tracing::trace_span!("scheduler.park").entered();
        if let Err(e) = self.executor.park_for_event(IDLE_LIVENESS_POLL) {
            self.on_executor_error(e);
            return;
        }
        // The death watcher wakes the park instantly on child exit; confirm and
        // latch it here (also the only death signal when fully idle, where no
        // result drain would otherwise surface it).
        if let Err(e) = self.executor.check_liveness() {
            self.on_executor_error(e);
        }
        self.stats
            .general
            .in_flight
            .store(self.executor.in_flight(), Ordering::Relaxed);
        self.publish_cache_stats();
    }
}
