use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::net::TcpListener;
use std::time::{Duration, Instant};

use anyhow::Context;
use uniserve_core::{CommandWaker, RequestId};
use uniserve_executor::{ControlAck, ControlOp, Executor, WorkerExecError, WorkerLossError};
use uniserve_worker_wire::{Batch, CompletionRecord, CompletionReport, EngineCaps, Point};

use crate::WorkerLaunchConfig;

/// How long a single rank may go without producing output, while batches are in
/// flight, before `next_result` treats it as dead and bails. Generous so a
/// healthy-but-slow forward is never falsely failed; short enough that a hung
/// rank does not wedge the scheduler loop forever.
const NEXT_RESULT_DEADLINE: Duration = Duration::from_secs(300);
/// Per-iteration bounded wait used while draining a lagging rank.
const NEXT_RESULT_POLL: Duration = Duration::from_millis(1);

#[derive(Clone)]
struct MultiprocSpawnSpec {
    python: String,
    model_dir: String,
    device: String,
    world_size: usize,
    pipeline_depth: usize,
    req_slot_cap: usize,
    resp_slot_cap: usize,
    kv_token_capacity: Option<u64>,
    block_size: u32,
    attention_backend: String,
    worker_kind: Option<String>,
    transfer_backend: Option<String>,
    defer_sampling: bool,
    worker_config: WorkerLaunchConfig,
}

impl MultiprocSpawnSpec {
    fn snapshots_enabled(&self) -> bool {
        self.worker_config.snapshot_dir.is_some()
    }

    fn launch(&self) -> anyhow::Result<Vec<Box<dyn Executor>>> {
        let tp_init_method = if self.world_size > 1 {
            Some(allocate_tp_init_method()?)
        } else {
            None
        };
        let mut launched = Vec::with_capacity(self.world_size);
        for rank in 0..self.world_size {
            let rank_device = device_for_rank(&self.device, rank, self.world_size);
            let mut rank_config = self.worker_config.clone();
            if let Some(root) = &self.worker_config.snapshot_dir {
                rank_config.snapshot_dir = Some(
                    std::path::PathBuf::from(root)
                        .join("ranks")
                        .join(rank.to_string())
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            launched.push(crate::UniprocExecutor::spawn_ranked_deferred_with_config(
                &self.python,
                &self.model_dir,
                &rank_device,
                self.pipeline_depth,
                self.req_slot_cap,
                self.resp_slot_cap,
                self.kv_token_capacity,
                self.block_size,
                &self.attention_backend,
                rank as u32,
                self.world_size as u32,
                tp_init_method.as_deref(),
                self.worker_kind.as_deref(),
                self.transfer_backend.as_deref(),
                self.defer_sampling,
                &rank_config,
            )?);
        }
        let mut workers: Vec<Box<dyn Executor>> = Vec::with_capacity(self.world_size);
        for mut worker in launched {
            worker.finish_startup()?;
            workers.push(Box::new(worker));
        }
        Ok(workers)
    }
}

/// W worker processes, one iceoryx2 request-response service each.
pub struct MultiprocExecutor {
    workers: Vec<Box<dyn Executor>>,
    buffers: Vec<VecDeque<CompletionReport>>,
    caps: EngineCaps,
    depth: usize,
    inflight: usize,
    next_call_id: u64,
    pending_batches: BTreeMap<u64, Batch>,
    spawn_spec: Option<MultiprocSpawnSpec>,
    known_sessions: BTreeSet<RequestId>,
}

impl MultiprocExecutor {
    pub fn new(workers: Vec<Box<dyn Executor>>) -> Self {
        Self::from_workers(workers, None)
    }

    fn from_workers(
        workers: Vec<Box<dyn Executor>>,
        spawn_spec: Option<MultiprocSpawnSpec>,
    ) -> Self {
        assert!(!workers.is_empty(), "need >= 1 worker");
        let n = workers.len();
        let mut caps = workers[0].caps();
        caps.rank.tp_size = n as u32;
        let depth = workers
            .iter()
            .map(|w| w.pipeline_depth())
            .min()
            .unwrap_or(1)
            .max(1);
        let buffers = (0..n).map(|_| VecDeque::new()).collect();
        Self {
            workers,
            buffers,
            caps,
            depth,
            inflight: 0,
            next_call_id: 1,
            pending_batches: BTreeMap::new(),
            spawn_spec,
            known_sessions: BTreeSet::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        python: &str,
        model_dir: &str,
        device: &str,
        world_size: usize,
        pipeline_depth: usize,
        req_slot_cap: usize,
        resp_slot_cap: usize,
        kv_token_capacity: Option<u64>,
        block_size: u32,
        attention_backend: &str,
    ) -> anyhow::Result<Self> {
        Self::spawn_with_config(
            python,
            model_dir,
            device,
            world_size,
            pipeline_depth,
            req_slot_cap,
            resp_slot_cap,
            kv_token_capacity,
            block_size,
            attention_backend,
            &WorkerLaunchConfig::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spawn_with_config(
        python: &str,
        model_dir: &str,
        device: &str,
        world_size: usize,
        pipeline_depth: usize,
        req_slot_cap: usize,
        resp_slot_cap: usize,
        kv_token_capacity: Option<u64>,
        block_size: u32,
        attention_backend: &str,
        worker_config: &WorkerLaunchConfig,
    ) -> anyhow::Result<Self> {
        Self::spawn_inner(
            python,
            model_dir,
            device,
            world_size,
            pipeline_depth,
            req_slot_cap,
            resp_slot_cap,
            kv_token_capacity,
            block_size,
            attention_backend,
            None,
            None,
            false,
            worker_config,
        )
    }

    /// Spawn a tensor-parallel pool for a staged worker kind: every rank is told
    /// which pipeline stage it serves. `world_size == 1` yields a one-rank pool
    /// (still a valid `Executor`), so the `StageRouter` composition can treat
    /// every pool uniformly regardless of its tp size.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_staged(
        python: &str,
        model_dir: &str,
        device: &str,
        world_size: usize,
        pipeline_depth: usize,
        req_slot_cap: usize,
        resp_slot_cap: usize,
        kv_token_capacity: Option<u64>,
        block_size: u32,
        attention_backend: &str,
        worker_kind: &str,
        transfer_backend: Option<&str>,
        defer_sampling: bool,
    ) -> anyhow::Result<Self> {
        Self::spawn_staged_with_config(
            python,
            model_dir,
            device,
            world_size,
            pipeline_depth,
            req_slot_cap,
            resp_slot_cap,
            kv_token_capacity,
            block_size,
            attention_backend,
            worker_kind,
            transfer_backend,
            defer_sampling,
            &WorkerLaunchConfig::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spawn_staged_with_config(
        python: &str,
        model_dir: &str,
        device: &str,
        world_size: usize,
        pipeline_depth: usize,
        req_slot_cap: usize,
        resp_slot_cap: usize,
        kv_token_capacity: Option<u64>,
        block_size: u32,
        attention_backend: &str,
        worker_kind: &str,
        transfer_backend: Option<&str>,
        defer_sampling: bool,
        worker_config: &WorkerLaunchConfig,
    ) -> anyhow::Result<Self> {
        Self::spawn_inner(
            python,
            model_dir,
            device,
            world_size,
            pipeline_depth,
            req_slot_cap,
            resp_slot_cap,
            kv_token_capacity,
            block_size,
            attention_backend,
            Some(worker_kind),
            transfer_backend,
            defer_sampling,
            worker_config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_inner(
        python: &str,
        model_dir: &str,
        device: &str,
        world_size: usize,
        pipeline_depth: usize,
        req_slot_cap: usize,
        resp_slot_cap: usize,
        kv_token_capacity: Option<u64>,
        block_size: u32,
        attention_backend: &str,
        worker_kind: Option<&str>,
        transfer_backend: Option<&str>,
        defer_sampling: bool,
        worker_config: &WorkerLaunchConfig,
    ) -> anyhow::Result<Self> {
        let spec = MultiprocSpawnSpec {
            python: python.to_string(),
            model_dir: model_dir.to_string(),
            device: device.to_string(),
            world_size: world_size.max(1),
            pipeline_depth,
            req_slot_cap,
            resp_slot_cap,
            kv_token_capacity,
            block_size,
            attention_backend: attention_backend.to_string(),
            worker_kind: worker_kind.map(str::to_string),
            transfer_backend: transfer_backend.map(str::to_string),
            defer_sampling,
            worker_config: worker_config.clone(),
        };
        let workers = spec.launch()?;
        Ok(Self::from_workers(workers, Some(spec)))
    }

    fn pump_once(&mut self) -> anyhow::Result<()> {
        for (i, worker) in self.workers.iter_mut().enumerate() {
            while let Some(result) = worker.poll()? {
                self.buffers[i].push_back(result);
            }
        }
        Ok(())
    }

    fn recover_workers(&mut self, cause: &anyhow::Error) -> anyhow::Result<()> {
        let mut spec = self.spawn_spec.clone().ok_or_else(|| {
            anyhow::anyhow!("worker process failed without a restart specification: {cause}")
        })?;
        spec.worker_config.restore_snapshots = spec.snapshots_enabled();
        tracing::warn!(error = %cause, "worker process lost; replacing its complete rank group");
        for worker in &mut self.workers {
            worker.shutdown();
        }

        let snapshots_enabled = spec.snapshots_enabled();
        let workers = match spec.launch() {
            Ok(workers) => workers,
            Err(restore_error) if snapshots_enabled => {
                let root =
                    spec.worker_config.snapshot_dir.clone().ok_or_else(|| {
                        anyhow::anyhow!("snapshot recovery requires a snapshot root")
                    })?;
                spec.worker_config.snapshot_dir = Some(
                    std::path::PathBuf::from(root)
                        .join("generations")
                        .join(format!("{:x}", crate::uniproc::nano_id()))
                        .to_string_lossy()
                        .into_owned(),
                );
                let workers = spec.launch().with_context(|| {
                    format!(
                        "worker replacement failed after snapshot restore error: {restore_error}"
                    )
                })?;
                self.install_replacement(workers, &spec)?;
                self.spawn_spec = Some(spec);
                self.discard_sessions_after_loss()?;
                return Err(WorkerLossError {
                    message: format!(
                        "worker snapshots could not be restored; affected sessions terminated: {restore_error}"
                    ),
                }
                .into());
            }
            Err(error) => return Err(error.context("spawning replacement worker ranks")),
        };
        self.install_replacement(workers, &spec)?;
        self.spawn_spec = Some(spec);

        let restored_by_rank = self.replacement_restored_sessions();
        let reconstructable = self.reconstructable_admissions();
        let required = self
            .known_sessions
            .difference(&reconstructable)
            .copied()
            .collect::<BTreeSet<_>>();
        if !snapshots_enabled
            || restored_by_rank
                .iter()
                .any(|restored| !required.is_subset(restored))
        {
            self.discard_sessions_after_loss()?;
            return Err(WorkerLossError {
                message: if snapshots_enabled {
                    "worker snapshot set was incomplete; affected sessions terminated".to_string()
                } else {
                    "worker process was replaced without snapshots; affected sessions terminated"
                        .to_string()
                },
            }
            .into());
        }

        for (rank, restored) in restored_by_rank.iter().enumerate() {
            for session_id in restored.difference(&self.known_sessions).copied() {
                let worker = &mut self.workers[rank];
                let acknowledgments =
                    worker.control_wait(ControlOp::DropSession(session_id), None)?;
                anyhow::ensure!(
                    acknowledgments.iter().all(|ack| ack.ok),
                    "replacement worker could not discard stale session {}",
                    session_id.0
                );
            }
        }
        for batch in self.pending_batches.values() {
            for (rank, worker) in self.workers.iter_mut().enumerate() {
                worker.submit(batch.clone()).with_context(|| {
                    format!(
                        "resubmitting step {} to replacement rank {rank}",
                        batch.step_id
                    )
                })?;
            }
        }
        self.inflight = self.pending_batches.len();
        tracing::info!(
            sessions = self.known_sessions.len(),
            batches = self.pending_batches.len(),
            "worker rank group restored from durable snapshots"
        );
        Ok(())
    }

    fn install_replacement(
        &mut self,
        workers: Vec<Box<dyn Executor>>,
        spec: &MultiprocSpawnSpec,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            workers.len() == spec.world_size,
            "replacement worker rank count changed"
        );
        for (rank, worker) in workers.iter().enumerate() {
            validate_replacement_caps(&self.caps, &worker.caps(), rank)?;
        }
        self.workers = workers;
        self.buffers = (0..spec.world_size).map(|_| VecDeque::new()).collect();
        Ok(())
    }

    fn replacement_restored_sessions(&self) -> Vec<BTreeSet<RequestId>> {
        self.workers
            .iter()
            .map(|worker| {
                worker
                    .caps()
                    .restored_sessions
                    .into_iter()
                    .collect::<BTreeSet<_>>()
            })
            .collect()
    }

    fn discard_sessions_after_loss(&mut self) -> anyhow::Result<()> {
        let restored_by_rank = self.replacement_restored_sessions();
        for (rank, restored) in restored_by_rank.into_iter().enumerate() {
            for session_id in restored {
                let worker = &mut self.workers[rank];
                let acknowledgments =
                    worker.control_wait(ControlOp::DropSession(session_id), None)?;
                anyhow::ensure!(
                    acknowledgments.iter().all(|ack| ack.ok),
                    "replacement worker could not discard unrestorable session {}",
                    session_id.0
                );
            }
        }
        self.pending_batches.clear();
        self.buffers.iter_mut().for_each(VecDeque::clear);
        self.inflight = 0;
        self.known_sessions.clear();
        Ok(())
    }

    fn reconstructable_admissions(&self) -> BTreeSet<RequestId> {
        self.pending_batches
            .values()
            .flat_map(|batch| {
                batch.operations.iter().filter_map(|operation| {
                    // A depth-one admission root: point zero of the request's
                    // admission operation. Only such operations can be replayed
                    // from a fresh admission after a rank group is replaced.
                    (matches!(operation.parent.point, Point::Fixed { point_index: 0, .. })
                        && batch
                            .admissions
                            .iter()
                            .any(|admission| admission.request_key == operation.request_key))
                    .then_some(operation.request_key.session_id)
                })
            })
            .collect()
    }

    fn discard_inflight(&mut self) {
        self.pending_batches.clear();
        self.buffers.iter_mut().for_each(VecDeque::clear);
        self.inflight = 0;
    }

    fn apply_control_session_effect(&mut self, operation: &ControlOp, succeeded: bool) {
        if !succeeded {
            return;
        }
        match operation {
            ControlOp::DropSession(session_id) => {
                self.known_sessions.remove(session_id);
            }
            ControlOp::RestoreSession(snapshot) => {
                self.known_sessions.insert(snapshot.session_id);
            }
            _ => {}
        }
    }

    fn pump(&mut self) -> anyhow::Result<()> {
        loop {
            match self.pump_once() {
                Ok(()) => return Ok(()),
                Err(error) if error.downcast_ref::<WorkerExecError>().is_some() => {
                    self.discard_inflight();
                    return Err(error);
                }
                Err(error) => {
                    self.recover_workers(&error)?;
                }
            }
        }
    }

    fn try_join(&mut self) -> anyhow::Result<Option<CompletionReport>> {
        if self.buffers.iter().any(|buffer| buffer.is_empty()) {
            return Ok(None);
        }
        let Some(step_id) = self.joinable_step_id() else {
            return Ok(None);
        };

        let mut per_rank = Vec::with_capacity(self.buffers.len());
        for buffer in &mut self.buffers {
            let pos = buffer
                .iter()
                .position(|report| report.step_id == step_id)
                .ok_or_else(|| {
                    anyhow::anyhow!("joinable step {step_id} disappeared from rank buffer")
                })?;
            per_rank.push(buffer.remove(pos).ok_or_else(|| {
                anyhow::anyhow!("joinable step {step_id} index disappeared from rank buffer")
            })?);
        }

        let mut out = per_rank.remove(0);
        for (rank, report) in per_rank.iter().enumerate() {
            merge_rank_report(step_id, &mut out, report, rank + 1)?;
        }
        out.worker_exec_us = per_rank
            .iter()
            .filter_map(|report| report.worker_exec_us)
            .chain(out.worker_exec_us)
            .max();
        self.inflight = self.inflight.saturating_sub(1);
        self.pending_batches.remove(&step_id);
        Ok(Some(out))
    }

    fn joinable_step_id(&self) -> Option<u64> {
        self.buffers[0]
            .iter()
            .filter_map(|report| {
                let step_id = report.step_id;
                self.buffers
                    .iter()
                    .all(|buffer| buffer.iter().any(|item| item.step_id == step_id))
                    .then_some(step_id)
            })
            .min()
    }
}

/// Join one step's per-rank completion reports into rank 0's. Tensor-parallel
/// ranks run the identical operation set, so every rank must report the same
/// semantic completion for each operation; only the per-rank product shards
/// differ and are concatenated in rank order.
fn merge_rank_report(
    step_id: u64,
    rank0: &mut CompletionReport,
    rankn: &CompletionReport,
    rank: usize,
) -> anyhow::Result<()> {
    if rankn.step_id != step_id {
        anyhow::bail!(
            "rank {rank} completion report step_id mismatch while joining step {step_id}: got {}",
            rankn.step_id
        );
    }
    if rankn.completions.len() != rank0.completions.len() {
        anyhow::bail!(
            "rank {rank} report for step {step_id} has {} completions, expected {}",
            rankn.completions.len(),
            rank0.completions.len()
        );
    }
    for (idx, (canonical, actual)) in rank0
        .completions
        .iter_mut()
        .zip(&rankn.completions)
        .enumerate()
    {
        if let Err(error) = merge_completion_record(canonical, actual) {
            anyhow::bail!(
                "rank {rank} report for step {step_id} completion {idx} differs from rank 0: {error:#}"
            );
        }
    }
    // Report-level product payloads are per-rank shards; concatenate them in
    // rank order (rank 0's, then rank 1's, ...).
    rank0.products.extend(rankn.products.iter().cloned());
    Ok(())
}

/// Merge one rank's completion record into rank 0's. Every rank must agree on
/// the semantic result; only `product_generations` are per-rank shards, which
/// concatenate in rank order.
fn merge_completion_record(
    canonical: &mut CompletionRecord,
    rank_completion: &CompletionRecord,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        canonical.request_key == rank_completion.request_key
            && canonical.op_id == rank_completion.op_id
            && canonical.selected_point == rank_completion.selected_point
            && canonical.semantic_digest == rank_completion.semantic_digest,
        "completion identity or selected point diverged"
    );
    anyhow::ensure!(
        canonical.committed_tokens == rank_completion.committed_tokens
            && canonical.status == rank_completion.status
            && canonical.logical_lengths == rank_completion.logical_lengths
            && canonical.token_span == rank_completion.token_span
            && canonical.finish_flags == rank_completion.finish_flags,
        "completion result fields diverged"
    );
    canonical
        .product_generations
        .extend(rank_completion.product_generations.iter().copied());
    Ok(())
}

fn validate_replacement_caps(
    expected: &EngineCaps,
    actual: &EngineCaps,
    rank: usize,
) -> anyhow::Result<()> {
    let mut normalized_expected = expected.clone();
    normalized_expected.rank.tp_rank = rank as u32;
    normalized_expected.restored_sessions = actual.restored_sessions.clone();
    anyhow::ensure!(
        &normalized_expected == actual,
        "replacement rank {rank} capabilities changed"
    );
    Ok(())
}

fn allocate_tp_init_method() -> anyhow::Result<String> {
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    let addr = listener.local_addr()?;
    Ok(format!("tcp://127.0.0.1:{}", addr.port()))
}

fn device_for_rank(device: &str, rank: usize, world_size: usize) -> String {
    let trimmed = device.trim();
    if world_size > 1 && matches!(trimmed, "cuda" | "gpu") {
        format!("cuda:{rank}")
    } else {
        device.to_string()
    }
}

impl Executor for MultiprocExecutor {
    fn caps(&self) -> EngineCaps {
        self.caps.clone()
    }

    fn pipeline_depth(&self) -> usize {
        self.depth
    }

    fn in_flight(&self) -> usize {
        self.inflight
    }

    fn submit(&mut self, batch: Batch) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.pending_batches.contains_key(&batch.step_id),
            "step {} is already in flight",
            batch.step_id
        );
        let step_id = batch.step_id;
        let sessions = batch
            .operations
            .iter()
            .map(|operation| operation.request_key.session_id)
            .collect::<Vec<_>>();
        self.pending_batches.insert(step_id, batch.clone());
        self.known_sessions.extend(sessions);
        self.inflight += 1;
        for (rank, worker) in self.workers.iter_mut().enumerate() {
            if let Err(error) = worker
                .submit(batch.clone())
                .with_context(|| format!("submit to rank {rank} failed"))
            {
                return self.recover_workers(&error);
            }
        }
        Ok(())
    }

    fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
        self.pump()?;
        self.try_join()
    }

    fn check_liveness(&mut self) -> anyhow::Result<()> {
        for (rank, worker) in self.workers.iter_mut().enumerate() {
            if let Err(error) = worker
                .check_liveness()
                .with_context(|| format!("rank {rank} liveness check failed"))
            {
                return self.recover_workers(&error);
            }
        }
        Ok(())
    }

    fn event_driven(&self) -> bool {
        // The pool is event-driven only when every rank is, so the scheduler's
        // single park covers them all (a mixed pool falls back to polling).
        !self.workers.is_empty() && self.workers.iter().all(|w| w.event_driven())
    }

    fn command_waker(&self) -> CommandWaker {
        // TP ranks run in lockstep, so rank 0's waker is enough to break the
        // park on a freshly enqueued command (the scheduler then drains all).
        self.workers
            .first()
            .map(|w| w.command_waker())
            .unwrap_or_else(CommandWaker::noop)
    }

    fn park_for_event(&mut self, timeout: std::time::Duration) -> anyhow::Result<()> {
        // Every rank receives the same batch and finishes within a hair of the
        // others, so parking on rank 0's wake surfaces a ready result without a
        // cross-endpoint WaitSet; a spurious wake is harmless because the
        // scheduler re-drains every rank after the park returns.
        match self.workers.first_mut() {
            Some(worker) => match worker.park_for_event(timeout) {
                Ok(()) => Ok(()),
                Err(error) => self.recover_workers(&error),
            },
            None => {
                std::thread::sleep(timeout.min(Duration::from_millis(1)));
                Ok(())
            }
        }
    }

    fn wait_result_timeout(
        &mut self,
        timeout: Duration,
    ) -> anyhow::Result<Option<CompletionReport>> {
        let Some(deadline) = Instant::now().checked_add(timeout) else {
            self.pump()?;
            return self.try_join();
        };
        loop {
            self.pump()?;
            if let Some(result) = self.try_join()? {
                return Ok(Some(result));
            }
            if self.inflight == 0 {
                return Ok(None);
            }
            let now = Instant::now();
            if now >= deadline {
                return Ok(None);
            }
            std::thread::sleep((deadline - now).min(Duration::from_millis(1)));
        }
    }

    fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
        if self.inflight == 0 {
            anyhow::bail!("next_result called with no in-flight batches");
        }
        // Bound the wait so a single dead/hung rank surfaces as an executor
        // error instead of wedging the scheduler loop forever. The deadline is
        // reset whenever any rank makes progress, so a healthy-but-slow batch is
        // never falsely failed.
        let mut deadline = Instant::now() + NEXT_RESULT_DEADLINE;
        loop {
            self.pump()?;
            if let Some(result) = self.try_join()? {
                return Ok(result);
            }
            // Wait for fresh output on whichever rank is currently behind. Using
            // the per-worker bounded wait (rather than the unbounded
            // next_result) keeps a hung rank from blocking indefinitely.
            let idx = self
                .buffers
                .iter()
                .position(|buffer| buffer.is_empty())
                .unwrap_or(0);
            match self.workers[idx].wait_result_timeout(NEXT_RESULT_POLL) {
                Ok(Some(result)) => {
                    self.buffers[idx].push_back(result);
                    deadline = Instant::now() + NEXT_RESULT_DEADLINE;
                }
                Ok(None) if Instant::now() >= deadline => {
                    let error = anyhow::anyhow!(
                        "rank {idx} produced no result within {:?} while {} batch(es) were in flight",
                        NEXT_RESULT_DEADLINE,
                        self.inflight
                    );
                    self.recover_workers(&error)?;
                    deadline = Instant::now() + NEXT_RESULT_DEADLINE;
                }
                Ok(None) => {}
                Err(error) if error.downcast_ref::<WorkerExecError>().is_some() => {
                    self.discard_inflight();
                    return Err(error);
                }
                Err(error) => {
                    self.recover_workers(&error)?;
                    deadline = Instant::now() + NEXT_RESULT_DEADLINE;
                }
            }
        }
    }

    fn control(&mut self, op: ControlOp) -> anyhow::Result<u64> {
        // NOTE: the returned id is a local fire-and-forget token, NOT a wire
        // call_id. Each per-rank worker.control allocates its own real
        // call_id internally; this counter correlates to none of them. Callers
        // must not use this value to match a later worker ack — use
        // control_wait (which fans out and collects per-rank acks) for any
        // call that needs correlation.
        let call_id = self.next_call_id;
        self.next_call_id += 1;
        loop {
            let mut failure = None;
            for rank in 0..self.workers.len() {
                if let Err(error) = self.workers[rank].control(op.clone()) {
                    failure = Some(error.context(format!("control call failed on rank {rank}")));
                    break;
                }
            }
            match failure {
                Some(error) => self.recover_workers(&error)?,
                None => break,
            }
        }
        self.apply_control_session_effect(&op, true);
        Ok(call_id)
    }

    fn control_wait(
        &mut self,
        op: ControlOp,
        targets: Option<&[u32]>,
    ) -> anyhow::Result<Vec<ControlAck>> {
        loop {
            let mut acks = Vec::new();
            let mut failure = None;
            for rank in 0..self.workers.len() {
                let rank = rank as u32;
                if let Some(targets) = targets
                    && !targets.contains(&rank)
                {
                    continue;
                }
                match self.workers[rank as usize].control_wait(op.clone(), None) {
                    Ok(rank_acks) => {
                        for mut ack in rank_acks {
                            ack.rank = rank;
                            acks.push(ack);
                        }
                    }
                    Err(error) => {
                        failure =
                            Some(error.context(format!("control wait failed on rank {rank}")));
                        break;
                    }
                }
            }
            if let Some(error) = failure {
                self.recover_workers(&error)?;
                continue;
            }
            self.apply_control_session_effect(&op, acks.iter().all(|ack| ack.ok));
            return Ok(acks);
        }
    }

    fn shutdown(&mut self) {
        for worker in self.workers.iter_mut() {
            worker.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::time::Duration;

    use super::{MultiprocExecutor, device_for_rank};
    use uniserve_core::RequestId;
    use uniserve_executor::{ControlAck, ControlOp, Executor};
    use uniserve_worker_wire::{
        Batch, CompletionRecord, CompletionReport, DType, EngineCaps, FinishFlags, LogicalLengths,
        OpId, OpStatus, PointRange, ProductKind, ProductPayload, ProductRef, RegistrationAck,
        RequestKey, ShapeBound, StorageClass, TimingCounters, TokenSpan,
    };

    #[test]
    fn cuda_device_is_ranked_for_multiproc() {
        assert_eq!(device_for_rank("cuda", 1, 4), "cuda:1");
        assert_eq!(device_for_rank("gpu", 2, 4), "cuda:2");
        assert_eq!(device_for_rank("cuda:0", 1, 4), "cuda:0");
        assert_eq!(device_for_rank("cpu", 1, 4), "cpu");
        assert_eq!(device_for_rank("cuda", 0, 1), "cuda");
    }

    #[test]
    fn multiproc_next_result_bails_with_no_inflight() {
        let mut exec = fake_multiproc(2);
        exec.inflight = 0;
        let err = exec.next_result().unwrap_err().to_string();
        assert!(err.contains("no in-flight batches"), "got: {err}");
    }

    #[test]
    fn multiproc_join_matches_by_step_id_without_consuming_unmatched_front() {
        let mut exec = fake_multiproc(2);
        exec.inflight = 2;
        exec.buffers[0].push_back(result(2, 20));
        exec.buffers[0].push_back(result(1, 10));
        exec.buffers[1].push_back(result(1, 10));

        let joined = exec.try_join().unwrap().unwrap();

        assert_eq!(joined.step_id, 1);
        assert_eq!(exec.inflight, 1);
        assert_eq!(exec.buffers[0].front().unwrap().step_id, 2);
        assert!(exec.buffers[1].is_empty());
    }

    #[test]
    fn multiproc_join_rejects_rank_token_divergence() {
        let mut exec = fake_multiproc(2);
        exec.inflight = 1;
        exec.buffers[0].push_back(result(1, 10));
        exec.buffers[1].push_back(result(1, 11));

        let err = exec.try_join().unwrap_err().to_string();

        assert!(err.contains("completion 0 differs"), "got: {err}");
        assert_eq!(exec.inflight, 1);
    }

    #[test]
    fn multiproc_join_concatenates_rank_ordered_products() {
        // The tensor-parallel join concatenates each completion's
        // `product_generations` and the report-level `products` in rank order,
        // while the semantic completion is required to be identical per rank.
        let mut exec = fake_multiproc(2);
        exec.inflight = 1;
        exec.buffers[0].push_back(result_with_products(1, vec![10, 11], 100));
        exec.buffers[1].push_back(result_with_products(1, vec![20, 21], 200));

        let joined = exec.try_join().unwrap().unwrap();

        assert_eq!(
            joined.completions[0].product_generations,
            vec![10, 11, 20, 21]
        );
        assert_eq!(joined.products.len(), 2);
        assert_eq!(joined.products[0].product.generation, 100);
        assert_eq!(joined.products[1].product.generation, 200);

        // Identity / committed-token divergence across ranks is still rejected.
        let mut exec = fake_multiproc(2);
        exec.inflight = 1;
        exec.buffers[0].push_back(result_with_products(1, vec![10], 100));
        let mut diverged = result_with_products(1, vec![20], 200);
        diverged.completions[0].committed_tokens = vec![99];
        exec.buffers[1].push_back(diverged);

        let error = exec.try_join().unwrap_err().to_string();
        assert!(error.contains("completion 0 differs"), "got: {error}");
    }

    fn fake_multiproc(n: usize) -> MultiprocExecutor {
        let workers = (0..n)
            .map(|_| Box::new(FakeExec::default()) as Box<dyn Executor>)
            .collect();
        MultiprocExecutor::new(workers)
    }

    fn completion(
        op: u64,
        committed_tokens: Vec<u32>,
        product_generations: Vec<u32>,
    ) -> CompletionRecord {
        CompletionRecord {
            request_key: RequestKey::new(1, RequestId(7), 1),
            op_id: OpId(op.max(1)),
            completion_slot_generation: 1,
            status: OpStatus::Ok,
            selected_point: 0,
            logical_lengths: LogicalLengths::default(),
            token_span: TokenSpan::default(),
            committed_tokens,
            finish_flags: FinishFlags::default(),
            product_generations,
            semantic_digest: "0".repeat(64),
            error_code: None,
            timing_counters: TimingCounters::default(),
        }
    }

    fn result(step_id: u64, sampled_token_id: u32) -> CompletionReport {
        CompletionReport {
            step_id,
            completions: vec![completion(step_id, vec![sampled_token_id], Vec::new())],
            products: Vec::new(),
            registration: RegistrationAck::default(),
            worker_exec_us: Some(step_id),
            forward_stats: None,
        }
    }

    fn result_with_products(
        step_id: u64,
        product_generations: Vec<u32>,
        payload_generation: u32,
    ) -> CompletionReport {
        CompletionReport {
            step_id,
            completions: vec![completion(step_id, vec![10], product_generations)],
            products: vec![product_payload(payload_generation)],
            registration: RegistrationAck::default(),
            worker_exec_us: Some(step_id),
            forward_stats: None,
        }
    }

    fn product_payload(generation: u32) -> ProductPayload {
        ProductPayload {
            product: ProductRef {
                request_key: RequestKey::new(1, RequestId(7), 1),
                producer_op_id: OpId(1),
                output_index: 0,
                generation,
                kind: ProductKind::Kv,
                storage_class: StorageClass::PagedKv,
                dtype: DType::BF16,
                shape_bound: ShapeBound::default(),
                point_range: PointRange::default(),
            },
            bytes: vec![generation as u8],
        }
    }

    #[derive(Default)]
    struct FakeExec {
        queued: VecDeque<CompletionReport>,
    }

    impl Executor for FakeExec {
        fn caps(&self) -> EngineCaps {
            EngineCaps::default()
        }

        fn pipeline_depth(&self) -> usize {
            2
        }

        fn in_flight(&self) -> usize {
            self.queued.len()
        }

        fn submit(&mut self, _batch: Batch) -> anyhow::Result<()> {
            Ok(())
        }

        fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
            Ok(self.queued.pop_front())
        }

        fn wait_result_timeout(
            &mut self,
            _timeout: Duration,
        ) -> anyhow::Result<Option<CompletionReport>> {
            self.poll()
        }

        fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
            self.queued
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("no fake result"))
        }

        fn control(&mut self, _op: ControlOp) -> anyhow::Result<u64> {
            Ok(0)
        }

        fn control_wait(
            &mut self,
            _op: ControlOp,
            _targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            Ok(Vec::new())
        }
    }
}
