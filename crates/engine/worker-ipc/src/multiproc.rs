use std::collections::VecDeque;
use std::net::TcpListener;
use std::time::{Duration, Instant};

use anyhow::Context;
use uniserve_core::CommandWaker;
use uniserve_executor::{ControlAck, ControlOp, Executor};
use uniserve_worker_wire::{EngineCaps, ForwardBatch, ForwardResult};

use crate::WorkerLaunchConfig;

/// How long a single rank may go without producing output, while batches are in
/// flight, before `next_result` treats it as dead and bails. Generous so a
/// healthy-but-slow forward is never falsely failed; short enough that a hung
/// rank does not wedge the scheduler loop forever.
const NEXT_RESULT_DEADLINE: Duration = Duration::from_secs(300);
/// Per-iteration bounded wait used while draining a lagging rank.
const NEXT_RESULT_POLL: Duration = Duration::from_millis(1);

/// W worker processes, one iceoryx2 request-response service each.
pub struct MultiprocExecutor {
    workers: Vec<Box<dyn Executor>>,
    buffers: Vec<VecDeque<ForwardResult>>,
    caps: EngineCaps,
    depth: usize,
    inflight: usize,
    next_call_id: u64,
}

impl MultiprocExecutor {
    pub fn new(workers: Vec<Box<dyn Executor>>) -> Self {
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
        let world_size = world_size.max(1);
        let tp_init_method = if world_size > 1 {
            Some(allocate_tp_init_method()?)
        } else {
            None
        };
        let mut launched = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            let rank_device = device_for_rank(device, rank, world_size);
            let worker = crate::UniprocExecutor::spawn_ranked_deferred_with_config(
                python,
                model_dir,
                &rank_device,
                pipeline_depth,
                req_slot_cap,
                resp_slot_cap,
                kv_token_capacity,
                block_size,
                attention_backend,
                rank as u32,
                world_size as u32,
                tp_init_method.as_deref(),
                worker_kind,
                transfer_backend,
                defer_sampling,
                worker_config,
            )?;
            launched.push(worker);
        }
        let mut workers: Vec<Box<dyn Executor>> = Vec::with_capacity(world_size);
        for mut worker in launched {
            worker.finish_startup()?;
            workers.push(Box::new(worker));
        }
        Ok(Self::new(workers))
    }

    fn pump(&mut self) -> anyhow::Result<()> {
        for (i, worker) in self.workers.iter_mut().enumerate() {
            while let Some(result) = worker.poll()? {
                self.buffers[i].push_back(result);
            }
        }
        Ok(())
    }

    fn try_join(&mut self) -> anyhow::Result<Option<ForwardResult>> {
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
                .position(|result| result.step_id == step_id)
                .ok_or_else(|| {
                    anyhow::anyhow!("joinable step {step_id} disappeared from rank buffer")
                })?;
            per_rank.push(buffer.remove(pos).ok_or_else(|| {
                anyhow::anyhow!("joinable step {step_id} index disappeared from rank buffer")
            })?);
        }

        let mut out = per_rank.remove(0);
        for (rank, result) in per_rank.iter().enumerate() {
            validate_rank_result(step_id, &out, result, rank + 1)?;
        }
        out.worker_exec_us = per_rank
            .iter()
            .filter_map(|result| result.worker_exec_us)
            .chain(out.worker_exec_us)
            .max();
        self.inflight = self.inflight.saturating_sub(1);
        Ok(Some(out))
    }

    fn joinable_step_id(&self) -> Option<u64> {
        self.buffers[0]
            .iter()
            .filter_map(|result| {
                let step_id = result.step_id;
                self.buffers
                    .iter()
                    .all(|buffer| buffer.iter().any(|item| item.step_id == step_id))
                    .then_some(step_id)
            })
            .min()
    }
}

fn validate_rank_result(
    step_id: u64,
    rank0: &ForwardResult,
    rankn: &ForwardResult,
    rank: usize,
) -> anyhow::Result<()> {
    if rankn.step_id != step_id {
        anyhow::bail!(
            "rank {rank} result step_id mismatch while joining step {step_id}: got {}",
            rankn.step_id
        );
    }
    if rankn.per_seq.len() != rank0.per_seq.len() {
        anyhow::bail!(
            "rank {rank} result for step {step_id} has {} seq results, expected {}",
            rankn.per_seq.len(),
            rank0.per_seq.len()
        );
    }
    for (idx, (a, b)) in rank0.per_seq.iter().zip(&rankn.per_seq).enumerate() {
        if a.req_id != b.req_id {
            anyhow::bail!(
                "rank {rank} result for step {step_id} seq {idx} has req_id {:?}, expected {:?}",
                b.req_id,
                a.req_id
            );
        }
        if a.sampled_token_id != b.sampled_token_id {
            anyhow::bail!(
                "rank {rank} result for step {step_id} req {:?} sampled token mismatch: {:?} != {:?}",
                a.req_id,
                b.sampled_token_id,
                a.sampled_token_id
            );
        }
        if a.num_accepted_tokens != b.num_accepted_tokens {
            anyhow::bail!(
                "rank {rank} result for step {step_id} req {:?} speculative acceptance mismatch: {:?} != {:?}",
                a.req_id,
                b.num_accepted_tokens,
                a.num_accepted_tokens
            );
        }
    }
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

    fn submit(&mut self, batch: ForwardBatch) -> anyhow::Result<()> {
        // `inflight` is bumped only after every rank has accepted the batch, so
        // a partial submit failure never corrupts the in-flight count. It does
        // leave the batch fanned out to ranks 0..rank while later ranks never
        // received it; the scheduler treats any submit error as fatal and tears
        // the engine down, so we surface which rank failed rather than attempt a
        // cross-rank rollback here.
        for (rank, worker) in self.workers.iter_mut().enumerate() {
            worker
                .submit(batch.clone())
                .with_context(|| format!("submit to rank {rank} failed"))?;
        }
        self.inflight += 1;
        Ok(())
    }

    fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>> {
        self.pump()?;
        self.try_join()
    }

    fn check_liveness(&mut self) -> anyhow::Result<()> {
        // any dead rank means the engine is dead. Probe every rank so an
        // idle-time exit of a single worker surfaces as a fatal liveness error.
        for (rank, worker) in self.workers.iter_mut().enumerate() {
            worker
                .check_liveness()
                .with_context(|| format!("rank {rank} liveness check failed"))?;
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
            Some(worker) => worker.park_for_event(timeout),
            None => {
                std::thread::sleep(timeout.min(Duration::from_millis(1)));
                Ok(())
            }
        }
    }

    fn wait_result_timeout(&mut self, timeout: Duration) -> anyhow::Result<Option<ForwardResult>> {
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

    fn next_result(&mut self) -> anyhow::Result<ForwardResult> {
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
            if let Some(result) = self.workers[idx].wait_result_timeout(NEXT_RESULT_POLL)? {
                self.buffers[idx].push_back(result);
                deadline = Instant::now() + NEXT_RESULT_DEADLINE;
            } else if Instant::now() >= deadline {
                anyhow::bail!(
                    "rank {idx} produced no result within {:?} while {} batch(es) were in flight",
                    NEXT_RESULT_DEADLINE,
                    self.inflight
                );
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
        for worker in self.workers.iter_mut() {
            worker.control(op.clone())?;
        }
        Ok(call_id)
    }

    fn control_wait(
        &mut self,
        op: ControlOp,
        targets: Option<&[u32]>,
    ) -> anyhow::Result<Vec<ControlAck>> {
        let mut acks = Vec::new();
        for (rank, worker) in self.workers.iter_mut().enumerate() {
            let rank = rank as u32;
            if let Some(targets) = targets
                && !targets.contains(&rank)
            {
                continue;
            }
            for mut ack in worker.control_wait(op.clone(), None)? {
                ack.rank = rank;
                acks.push(ack);
            }
        }
        Ok(acks)
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
    use uniserve_worker_wire::{EngineCaps, ForwardBatch, ForwardResult, SeqResult};

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

        assert!(err.contains("sampled token mismatch"));
        assert_eq!(exec.inflight, 1);
    }

    fn fake_multiproc(n: usize) -> MultiprocExecutor {
        let workers = (0..n)
            .map(|_| Box::new(FakeExec::default()) as Box<dyn Executor>)
            .collect();
        MultiprocExecutor::new(workers)
    }

    fn result(step_id: u64, sampled_token_id: u32) -> ForwardResult {
        ForwardResult {
            step_id,
            per_seq: vec![SeqResult {
                req_id: RequestId(7),
                sampled_token_id: Some(sampled_token_id),
                op_id: Some(step_id),
                ..SeqResult::default()
            }],
            worker_exec_us: Some(step_id),
            forward_stats: None,
        }
    }

    #[derive(Default)]
    struct FakeExec {
        queued: VecDeque<ForwardResult>,
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

        fn submit(&mut self, _batch: ForwardBatch) -> anyhow::Result<()> {
            Ok(())
        }

        fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>> {
            Ok(self.queued.pop_front())
        }

        fn wait_result_timeout(
            &mut self,
            _timeout: Duration,
        ) -> anyhow::Result<Option<ForwardResult>> {
            self.poll()
        }

        fn next_result(&mut self) -> anyhow::Result<ForwardResult> {
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
