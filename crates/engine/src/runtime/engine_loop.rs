use super::*;

impl EngineLoop {
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
        config: SchedulerConfig,
    ) -> Self {
        let info = executor
            .info()
            .runtime_info()
            .expect("executor exposes a valid runtime capacity view");
        let work = &info.supported_ops;
        let family = if work.contains(&OpKind::DiffusionPrepare)
            && !work.contains(&OpKind::ArDecode)
        {
            RuntimeFamily::Diffusion
        } else if work.contains(&OpKind::DiffusionStep) || work.contains(&OpKind::EncoderExecute) {
            RuntimeFamily::Umm
        } else {
            RuntimeFamily::Ar
        };
        Self::with_config_for_family(executor, ctrl, config, family)
    }

    pub fn with_config_for_family(
        executor: Box<dyn Executor>,
        ctrl: ControlTokens,
        config: SchedulerConfig,
        family: RuntimeFamily,
    ) -> Self {
        let profile = match family {
            RuntimeFamily::Ar => RuntimeProfile::ar(uniserve_core::ModelDtype::BFloat16),
            RuntimeFamily::Diffusion => {
                RuntimeProfile::diffusion(uniserve_core::ModelDtype::BFloat16)
            }
            RuntimeFamily::Umm => RuntimeProfile::umm(
                uniserve_core::ModelDtype::BFloat16,
                sim_umm_generation_limits(),
            ),
        };
        Self::with_runtime_profile(executor, ctrl, config, family, profile)
    }

    pub fn with_runtime_profile(
        executor: Box<dyn Executor>,
        ctrl: ControlTokens,
        mut config: SchedulerConfig,
        family: RuntimeFamily,
        profile: RuntimeProfile,
    ) -> Self {
        let info = executor
            .info()
            .runtime_info()
            .expect("executor exposes a valid runtime capacity view");
        let profile = profile.resolved(&info);
        let max_batch_ops = info.max_batch_ops as usize;
        let max_batch_tokens = info.max_batch_tokens as usize;
        let transfer_capacity = (info.queue_depth as usize)
            .saturating_mul(max_batch_ops)
            .clamp(1, MAX_INFLIGHT_TRANSFERS);
        config.max_num_waiting = config.max_num_waiting.clamp(1, MAX_NUM_WAITING);
        let flow_slot_reserve =
            usize::from(info.uses_kv() && info.supported_ops.contains(&OpKind::DiffusionStep));
        let request_pool_capacity = info.request_slots as usize;
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
        let kv = worker_kv_state(&info);
        let stats = Arc::new(SchedStats::default());
        stats.kv_cache.num_blocks.store(
            kv.as_ref().map_or(0, |state| state.usable_blocks),
            Ordering::Relaxed,
        );
        let memory = Memory::with_buffer_capacity(
            &info,
            info.buffer_pool_bytes,
            profile.encoder_cache_entries,
        );
        let denoise_step_burst = denoise_step_burst_from_env();
        let flow_exclusive_batch = env::var(FLOW_EXCLUSIVE_BATCH_ENV)
            .is_ok_and(|raw| matches!(raw.trim(), "1" | "true" | "TRUE"));
        let mut trace_sink = crate::runtime::bench_trace::RuntimeTraceSink::from_env();
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
                    "block_size": info.kv_block_size(),
                    "num_blocks": info.kv_num_blocks(),
                    "supported_ops": &info.supported_ops,
                    "max_batch_ops": info.max_batch_ops,
                    "max_batch_tokens": info.max_batch_tokens,
                    "request_slots": info.request_slots,
                    "queue_depth": info.queue_depth,
                    "latent_page_units": info.latent_page_units,
                    "latent_pages": info.latent_pages,
                    "latent_dtype": &profile.latent_dtype,
                    "latent_downsample": profile.generation_limits.latent_downsample,
                    "max_vae_grid_tokens": profile.generation_limits.max_vae_grid_tokens,
                    "max_vit_grid_tokens": profile.generation_limits.max_vit_grid_tokens,
                    "commit_marker_tokens": profile.generation_limits.commit_marker_tokens,
                    "max_cfg_branches": profile.generation_limits.max_cfg_branches,
                },
            }));
        }
        let latent_dtype = profile.latent_dtype;
        let runtime = Runtime::new(
            family,
            RuntimeState {
                ctrl,
                logits_pipeline: crate::runtime::logits::default_pipeline(),
                waiting: HashMap::new(),
                waiting_media: HashMap::new(),
                running: HashMap::new(),
                running_media: HashMap::new(),
                retiring_requests: HashMap::new(),
                retiring_media: HashMap::new(),
                inflight: InflightWindow::new(transfer_capacity),
                denoise_step_burst,
                latent_dtype,
                pending_commands: VecDeque::new(),
                pending_buffer_frees: HashMap::new(),
                authority_id: 1,
                next_op_id: 1,
                next_product_generation: 1,
                next_epoch: 1,
            },
        );
        Self {
            executor,
            pending_submission: None,
            info,
            profile,
            memory,
            runtime,
            scheduler: Scheduler::new(config),
            flow_exclusive_batch,
            fatal: false,
            trace_sink,
            peak_ops_in_batch: 0,
            stats,
        }
    }

    pub fn policy(&self) -> SchedulingPolicy {
        self.scheduler.config.policy
    }
    pub fn config(&self) -> &SchedulerConfig {
        &self.scheduler.config
    }
    pub fn set_prefix_cache(&mut self, on: bool) {
        self.memory.set_prefix_cache(on);
    }
    pub fn set_hash_algo(&mut self, algo: HashAlgo) {
        self.memory.set_hash_algo(algo);
    }
    /// Configure the per-step token budget.
    pub fn set_token_budget(&mut self, tokens: usize) {
        self.scheduler.config.max_num_batched_tokens = tokens.max(1);
    }
    pub fn set_long_prefill_threshold(&mut self, n: usize) {
        self.scheduler.config.long_prefill_threshold = n.max(1);
    }
    pub fn set_max_num_seqs(&mut self, n: usize) {
        let flow_slot_reserve = usize::from(
            self.info.uses_kv() && self.info.supported_ops.contains(&OpKind::DiffusionStep),
        );
        let capacity = self
            .memory
            .request_slots
            .capacity()
            .saturating_sub(flow_slot_reserve)
            .max(1);
        self.scheduler.config.max_num_seqs = n.clamp(1, capacity);
    }
    /// Cap waiting and terminal-output-retained request state.
    pub fn set_max_num_waiting(&mut self, n: usize) {
        self.scheduler.config.max_num_waiting = n.clamp(1, MAX_NUM_WAITING);
    }
    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }
    pub fn runtime_profile(&self) -> &RuntimeProfile {
        &self.profile
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
        self.scheduler
            .waiting_len()
            .saturating_add(self.scheduler.waiting_media_len())
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

    /// One park over result, command, worker-death, CPU-continuation, and
    /// output-capacity wakes. The timeout is solely a liveness deadline.
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
            .store(self.inflight.batch_started.len(), Ordering::Relaxed);
        self.publish_cache_stats();
    }
}
