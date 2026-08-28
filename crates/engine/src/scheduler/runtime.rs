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
        let info = executor.info().clone();
        let max_batch_ops = info.max_batch_operations as usize;
        let max_batch_tokens = info.max_batch_tokens as usize;
        let transfer_capacity = (info.pipeline_depth as usize)
            .saturating_mul(max_batch_ops)
            .clamp(1, MAX_INFLIGHT_TRANSFERS);
        config.max_num_waiting = config.max_num_waiting.clamp(1, MAX_NUM_WAITING);
        let flow_slot_reserve =
            usize::from(info.uses_kv() && info.supported_work.contains(&ForwardMode::GenFlow));
        let request_pool_capacity = info.max_request_pool_size as usize;
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
        let kv = info.uses_kv().then(|| {
            let block_pool = if info.groups.is_empty() {
                BlockPool::new(info.num_blocks as usize, info.block_size as usize)
            } else {
                let mut offset = 0_u32;
                let group_shapes: Vec<(uniserve_core::KvGroupKind, u32, u32)> = info
                    .groups
                    .iter()
                    .map(|group| {
                        let group_shape = (group.kind, offset, group.num_blocks);
                        offset = offset.saturating_add(group.num_blocks);
                        group_shape
                    })
                    .collect();
                BlockPool::with_groups(
                    info.num_blocks as usize,
                    info.block_size as usize,
                    &group_shapes,
                )
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
        let encoder_budget = info.encoder_cache_budget as usize;
        let kv_budget = KvBudget::new(
            kv,
            encoder_budget,
            request_pool_capacity,
            info.num_latent_pages,
            info.latent_page_units,
        );
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
                "info": {
                    "block_size": info.block_size,
                    "num_blocks": info.num_blocks,
                    "supported_work": &info.supported_work,
                    "max_batch_operations": info.max_batch_operations,
                    "max_batch_tokens": info.max_batch_tokens,
                    "max_request_pool_size": info.max_request_pool_size,
                    "pipeline_depth": info.pipeline_depth,
                    "latent_page_units": info.latent_page_units,
                    "num_latent_pages": info.num_latent_pages,
                    "latent_width": info.latent_width,
                    "latent_dtype": &info.latent_dtype,
                    "latent_downsample": info.latent_downsample,
                    "max_vae_grid_tokens": info.max_vae_grid_tokens,
                    "max_vit_grid_tokens": info.max_vit_grid_tokens,
                    "commit_marker_tokens": info.commit_marker_tokens,
                    "gen_rope_advance": info.gen_rope_advance,
                    "max_cfg_branches": info.max_cfg_branches,
                    "resource_classes": &info.resource_classes,
                },
            }));
        }
        let latent_dtype = worker_float_dtype(info.latent_dtype);
        Self {
            executor,
            info,
            kv_budget,
            ctrl,
            pending: RequestQueue::new(config.policy),
            config,
            logits_pipeline: crate::scheduler::logits::default_pipeline(),
            running: HashMap::new(),
            running_media: HashMap::new(),
            output: OutputSender::default(),
            retiring_sessions: HashMap::new(),
            order: Vec::new(),
            pending_media: VecDeque::new(),
            retiring_media: HashMap::new(),
            prefer_media: true,
            inflight: InflightWindow::new(transfer_capacity),
            denoise_step_burst,
            flow_exclusive_batch,
            fatal: false,
            latent_dtype,
            pending_controls: VecDeque::new(),
            authority_id: 1,
            next_op_id: 1,
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
        self.kv_budget.set_prefix_cache(on);
    }
    pub fn set_hash_algo(&mut self, algo: HashAlgo) {
        self.kv_budget.set_hash_algo(algo);
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
            self.info.uses_kv() && self.info.supported_work.contains(&ForwardMode::GenFlow),
        );
        let capacity = self
            .kv_budget
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
    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }
    pub fn stats_handle(&self) -> Arc<SchedStats> {
        self.stats.clone()
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
