//! Control-plane stage routing + data-plane Tier 1.
//!
//! [`StageRouter`] generalizes fixed 2-role (Understanding/Generation) splits
//! to an N-pool router keyed by [`OpKind`]: it partitions a single
//! [`ForwardBatch`] across the pools that own each op's kind, then merges the
//! per-pool [`SeqResult`]s back into the original op order before returning a
//! [`ForwardResult`] to the scheduler. It implements the [`Executor`] trait, so
//! the scheduler drives it exactly like a single pool.
//!
//! [`StageRouter::two_role`] is the Understanding/Generation 2-pool constructor.
//!
//! [`TensorMover`] is the data-plane Tier 1 (handle semantics) layered over the
//! pluggable Tier-2 [`TransferAgent`](uniserve_worker_ipc_core::TransferAgent).
//! With the in-process data plane (direct-reference, zero transfer), the mover
//! initiates no byte movement and reports every step's transfers complete, so a
//! single-pool `StageRouter` is behavior-identical to the underlying executor.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use uniserve_core::{CommandWaker, RequestId};
use uniserve_executor::{ControlAck, ControlOp, Executor, WorkerKind};
use uniserve_worker_wire::{
    EngineCaps, ForwardBatch, ForwardOp, ForwardResult, NewRequestData, OpKind, SeqResult,
};

/// Every op kind the wire defines. Used to build a single-pool routing table
/// (all kinds → pool 0). Kept exhaustive via the match in [`all_op_kinds`].
fn all_op_kinds() -> [OpKind; 10] {
    // A match (not a literal) so adding an OpKind variant fails to compile here
    // until it is classified, keeping routing tables exhaustive.
    fn _exhaustive(k: OpKind) {
        match k {
            OpKind::PrefillUnd
            | OpKind::DecodeUnd
            | OpKind::TargetVerifyUnd
            | OpKind::DenoiseGen
            | OpKind::CommitGen
            | OpKind::CommitWriteback
            | OpKind::VaeEncode
            | OpKind::VitEncode
            | OpKind::Sample
            | OpKind::EncodeFrame => {}
        }
    }
    [
        OpKind::PrefillUnd,
        OpKind::DecodeUnd,
        OpKind::TargetVerifyUnd,
        OpKind::DenoiseGen,
        OpKind::CommitGen,
        OpKind::CommitWriteback,
        OpKind::VaeEncode,
        OpKind::VitEncode,
        OpKind::Sample,
        OpKind::EncodeFrame,
    ]
}

/// Tier-1 data-plane mover: tracks per-request and per-step cross-pool transfer
/// completion for the scheduler's `stage_ready` gate.
///
/// The host never moves bytes; the actual copy is worker-side over the registered
/// data plane (`cuda_ipc`/`mooncake`). This tracker records the causal dependency:
/// a consumer-pool op (e.g. the gen tower's `denoise_gen`) must not dispatch until
/// the producer's conditioning-KV locator has arrived and been threaded into it.
/// A request with no recorded edge — the non-disaggregated single-pool /
/// decode→sampler path — is always ready, so the default topology is unaffected.
#[derive(Default)]
pub struct TensorMover {
    /// req → conditioning-KV locator the gen pool must fetch for its denoise op.
    /// Present iff the und pool published it and it has not yet been threaded.
    conditioning: HashMap<RequestId, String>,
    /// req → generated latent locator the und pool must fetch for writeback.
    commit_latent: HashMap<RequestId, String>,
}

impl TensorMover {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the und pool's conditioning-KV locator for `req` (read-driven, §4.2):
    /// the gen pool's denoise op fetches it synchronously, so this only stores the
    /// locator for threading; it never gates a step or a request (gating either
    /// would deadlock the producer step or the consumer op that clears it).
    pub fn record_conditioning(&mut self, req: RequestId, locator: String, _step_id: u64) {
        self.conditioning.insert(req, locator);
    }

    /// Take the recorded conditioning locator to thread into the gen pool's denoise
    /// op (once). Returns `None` if none is pending.
    pub fn take_conditioning(&mut self, req: RequestId) -> Option<String> {
        self.conditioning.remove(&req)
    }

    pub fn record_commit_latent(&mut self, req: RequestId, locator: String) {
        self.commit_latent.insert(req, locator);
    }

    pub fn take_commit_latent(&mut self, req: RequestId) -> Option<String> {
        self.commit_latent.remove(&req)
    }

    /// Whether a request's next op may be dispatched w.r.t. data-plane causality.
    /// Read-driven fetch is synchronous in the consumer op, so always ready; the
    /// non-disaggregated path records nothing and is likewise unaffected.
    pub fn request_ready(&self, _req_id: RequestId) -> bool {
        true
    }
}

/// One pool participating in the stage topology, with its declared role.
struct PoolEntry {
    kind: WorkerKind,
    exec: Box<dyn Executor>,
}

struct PendingStep {
    /// Bitmask of pool indices that still owe a result for this step.
    expected: u64,
    /// Pool index each original op routed to (parallel to the merged output).
    routes: Vec<usize>,
    /// Per-original-op merged result buffer.
    outputs: Vec<Option<SeqResult>>,
    /// Synthesized `sample` ops still outstanding at the Sampler pool for this
    /// step (the decode→sampler second phase). The step finalizes only once
    /// `expected == 0` AND `samples_pending == 0`.
    samples_pending: u64,
    worker_exec_us: u64,
}

/// N-pool, OpKind-routed executor. See module docs.
pub struct StageRouter {
    routing: HashMap<OpKind, usize>,
    pools: Vec<PoolEntry>,
    caps: EngineCaps,
    depth: usize,
    pending: BTreeMap<u64, PendingStep>,
    ready: VecDeque<ForwardResult>,
    next_call_id: u64,
    mover: TensorMover,
    /// Index of the Sampler pool, if the topology peels sampling. When set, any
    /// result carrying a `logits_handle` triggers a synthesized `sample` op
    /// routed here; the sampled token is backfilled.
    sampler_pool: Option<usize>,
    /// The und→gen tower-disaggregation edge `(und_pool, gen_pool)`, present only
    /// when the two towers are distinct pools (Mode A). When set, a conditioning
    /// locator on an und-pool result is recorded and threaded into the gen pool's
    /// `denoise_gen` op for that request (cross-step, §4.2).
    tower_edge: Option<(usize, usize)>,
    /// The gen→und commit edge `(gen_pool, und_pool)`, present only for Mode A.
    commit_edge: Option<(usize, usize)>,
    /// Per-request sampling descriptor, remembered from `new_reqs` so a
    /// synthesized `sample` op can carry the request's `SamplingParams` to the
    /// Sampler (which never saw the original first-dispatch `new_reqs`).
    req_sampling: HashMap<RequestId, NewRequestData>,
    /// Synthesized-sample op_id → (step_id, output slot) for backfill correlation.
    pending_samples: HashMap<u64, (u64, usize)>,
    next_sample_op_id: u64,
}

/// Append items from `incoming` not already in `target`, preserving order, with
/// O(n) membership via a HashSet of the values seen so far.
fn extend_unique<T: Clone + Eq + std::hash::Hash>(target: &mut Vec<T>, incoming: Vec<T>) {
    let mut seen: HashSet<T> = target.iter().cloned().collect();
    for item in incoming {
        if seen.insert(item.clone()) {
            target.push(item);
        }
    }
}

impl StageRouter {
    /// Build a router from `(WorkerKind, Executor)` pools. Each pool's
    /// `WorkerKind::supported_ops` defines which op kinds route to it. A kind
    /// claimed by two pools routes to the first (declaration order).
    #[cfg(test)]
    fn new(pools: Vec<(WorkerKind, Box<dyn Executor>)>) -> Self {
        Self::try_new(pools).expect("invalid StageRouter pool capabilities")
    }

    /// Build a staged router after validating the cross-pool capability contract.
    pub fn try_new(pools: Vec<(WorkerKind, Box<dyn Executor>)>) -> anyhow::Result<Self> {
        anyhow::ensure!(!pools.is_empty(), "StageRouter needs at least one pool");
        let mut routing: HashMap<OpKind, usize> = HashMap::new();
        for (idx, (kind, exec)) in pools.iter().enumerate() {
            let caps = exec.caps();
            for op in kind.supported_ops() {
                if caps.supported_ops.contains(op) {
                    routing.entry(*op).or_insert(idx);
                }
            }
        }
        let pools: Vec<PoolEntry> = pools
            .into_iter()
            .map(|(kind, exec)| PoolEntry { kind, exec })
            .collect();
        let caps = Self::merge_caps(&pools, &routing)?;
        let depth = pools
            .iter()
            .map(|p| p.exec.pipeline_depth())
            .min()
            .unwrap_or(1)
            .max(1);
        let sampler_pool = pools.iter().position(|p| p.kind == WorkerKind::Sampler);
        // The und→gen tower-disaggregation edge: the pool that produces the
        // conditioning KV (handles `decode_und`) and the pool that consumes it
        // (handles `denoise_gen`). The edge is active only when they are distinct
        // pools (Mode A); a single full pool has `und_pool == gen_pool` and no
        // cross-pool crossing, so the threading below is inert.
        let und_pool = routing.get(&OpKind::DecodeUnd).copied();
        let gen_pool = routing.get(&OpKind::DenoiseGen).copied();
        let tower_edge = match (und_pool, gen_pool) {
            (Some(u), Some(g)) if u != g => Some((u, g)),
            _ => None,
        };
        let commit_edge = tower_edge.map(|(u, g)| (g, u));
        Ok(Self {
            routing,
            pools,
            caps,
            depth,
            pending: BTreeMap::new(),
            ready: VecDeque::new(),
            next_call_id: 1,
            mover: TensorMover::new(),
            sampler_pool,
            tower_edge,
            commit_edge,
            req_sampling: HashMap::new(),
            pending_samples: HashMap::new(),
            // Synthesized sample op_ids live in a high range so they never
            // collide with scheduler-assigned op_ids on the wire echo.
            next_sample_op_id: 1 << 56,
        })
    }

    fn merge_caps(
        pools: &[PoolEntry],
        routing: &HashMap<OpKind, usize>,
    ) -> anyhow::Result<EngineCaps> {
        let routed_caps = |kind: OpKind| routing.get(&kind).map(|index| pools[*index].exec.caps());
        let mut kv_pool_indices = [
            OpKind::PrefillUnd,
            OpKind::DecodeUnd,
            OpKind::TargetVerifyUnd,
            OpKind::CommitWriteback,
        ]
        .into_iter()
        .filter_map(|kind| routing.get(&kind).copied())
        .collect::<Vec<_>>();
        kv_pool_indices.sort_unstable();
        kv_pool_indices.dedup();
        let seed_index = kv_pool_indices.first().copied().unwrap_or(0);
        let mut acc = pools[seed_index].exec.caps();

        if let Some(first_index) = kv_pool_indices.first().copied() {
            let first = pools[first_index].exec.caps();
            for index in kv_pool_indices.iter().copied().skip(1) {
                let other = pools[index].exec.caps();
                anyhow::ensure!(
                    other.block_size == first.block_size,
                    "staged KV pools disagree on block_size: {} versus {}",
                    first.block_size,
                    other.block_size
                );
                anyhow::ensure!(
                    other.num_layers == first.num_layers
                        && other.kv_dtype == first.kv_dtype
                        && other.quantization == first.quantization
                        && other.groups == first.groups,
                    "staged KV pools expose incompatible cache layouts"
                );
            }
            acc.block_size = first.block_size;
            acc.num_blocks = kv_pool_indices
                .iter()
                .map(|index| pools[*index].exec.caps().num_blocks)
                .min()
                .unwrap_or(first.num_blocks);
            acc.num_layers = first.num_layers;
            acc.groups = first.groups;
            acc.kv_dtype = first.kv_dtype;
            acc.quantization = first.quantization;
            acc.bytes_per_token = kv_pool_indices
                .iter()
                .map(|index| pools[*index].exec.caps().bytes_per_token)
                .max()
                .unwrap_or(first.bytes_per_token);
        }

        acc.supported_ops = all_op_kinds()
            .into_iter()
            .filter(|kind| routing.contains_key(kind))
            .collect();
        acc.supported_controls.clear();
        acc.resource_classes.clear();
        for pool in pools {
            let caps = pool.exec.caps();
            extend_unique(&mut acc.supported_controls, caps.supported_controls);
            extend_unique(&mut acc.resource_classes, caps.resource_classes);
        }
        acc.pipeline_depth = pools
            .iter()
            .map(|pool| pool.exec.caps().pipeline_depth.max(1))
            .min()
            .unwrap_or(1);
        let batch_limits = pools
            .iter()
            .map(|pool| pool.exec.caps().execution_constraints.max_batch_ops)
            .filter(|limit| *limit > 0)
            .collect::<Vec<_>>();
        acc.execution_constraints.max_batch_ops = batch_limits.into_iter().min().unwrap_or(0);

        let denoise = routed_caps(OpKind::DenoiseGen);
        acc.max_latent_size = denoise.as_ref().map_or(0, |caps| caps.max_latent_size);
        acc.latent_downsample = denoise.as_ref().map_or(0, |caps| caps.latent_downsample);
        acc.max_cfg_branches = denoise.as_ref().map_or(0, |caps| caps.max_cfg_branches);
        acc.scratch_capacity_tokens = denoise
            .as_ref()
            .map_or(0, |caps| caps.scratch_capacity_tokens);
        acc.max_vae_grid_tokens = routed_caps(OpKind::VaeEncode)
            .as_ref()
            .map_or(0, |caps| caps.max_vae_grid_tokens);
        acc.max_vit_grid_tokens = routed_caps(OpKind::VitEncode)
            .as_ref()
            .map_or(0, |caps| caps.max_vit_grid_tokens);
        let commit = routed_caps(OpKind::CommitGen);
        acc.commit_marker_tokens = commit
            .as_ref()
            .map_or(acc.commit_marker_tokens, |caps| caps.commit_marker_tokens);
        acc.gen_rope_advance = commit
            .as_ref()
            .map_or(acc.gen_rope_advance, |caps| caps.gen_rope_advance);
        acc.encoder_cache_budget = [OpKind::VaeEncode, OpKind::VitEncode]
            .into_iter()
            .filter_map(|kind| routed_caps(kind).map(|caps| caps.encoder_cache_budget))
            .min()
            .unwrap_or(0);
        if let Some(decode) = routed_caps(OpKind::DecodeUnd) {
            acc.adapter_mode = decode.adapter_mode;
        }
        Ok(acc)
    }

    fn route_for(&self, op: &ForwardOp) -> anyhow::Result<usize> {
        self.routing.get(&op.kind).copied().ok_or_else(|| {
            anyhow::anyhow!(
                "StageRouter has no pool for op kind {:?} (req {:?})",
                op.kind,
                op.req_id
            )
        })
    }

    fn pool_bit(idx: usize) -> u64 {
        1u64 << idx
    }

    fn submit_partition(
        exec: &mut dyn Executor,
        step_id: u64,
        new_reqs: Vec<NewRequestData>,
        ops: Vec<ForwardOp>,
    ) -> anyhow::Result<bool> {
        if ops.is_empty() {
            return Ok(false);
        }
        exec.submit(ForwardBatch {
            step_id,
            new_reqs,
            ops,
        })?;
        Ok(true)
    }

    fn pump(&mut self) -> anyhow::Result<()> {
        for idx in 0..self.pools.len() {
            while let Some(result) = self.pools[idx].exec.poll()? {
                self.route_result(idx, result)?;
            }
        }
        Ok(())
    }

    fn route_result(&mut self, pool_idx: usize, result: ForwardResult) -> anyhow::Result<()> {
        // A result from the Sampler pool is the second phase of a decode→sampler
        // handoff (synthesized `sample` ops), not an original-op partition; route
        // it to backfill by op_id rather than by slot.
        if self.sampler_pool == Some(pool_idx) {
            return self.backfill_samples(result);
        }
        let step_id = result.step_id;
        let bit = Self::pool_bit(pool_idx);
        // First pass under an immutable-ish borrow: validate + collect the
        // sample-synthesis requests (ops whose worker deferred sampling, i.e.
        // returned a `logits_handle`), then fill the slots.
        let mut to_sample: Vec<(usize, RequestId, Option<u64>, Option<String>)> = Vec::new();
        // Conditioning-KV locators an und pool published, to record for threading
        // into the matching `denoise_gen` op (Mode A und→gen edge).
        let mut to_record: Vec<(RequestId, String)> = Vec::new();
        let mut to_record_commit: Vec<(RequestId, String)> = Vec::new();
        {
            let Some(step) = self.pending.get_mut(&step_id) else {
                anyhow::bail!("stage pool returned unknown step_id {step_id}");
            };
            if step.expected & bit == 0 {
                anyhow::bail!(
                    "duplicate or unexpected pool {pool_idx} result for step_id {step_id}"
                );
            }
            let slots: Vec<usize> = step
                .outputs
                .iter()
                .enumerate()
                .filter_map(|(i, slot)| (slot.is_none() && step.routes[i] == pool_idx).then_some(i))
                .collect();
            if slots.len() != result.per_seq.len() {
                anyhow::bail!(
                    "pool {pool_idx} result count mismatch for step_id {step_id}: expected {}, got {}",
                    slots.len(),
                    result.per_seq.len()
                );
            }
            let sampler_split = self.sampler_pool.is_some();
            let und_pool = self.tower_edge.map(|(u, _)| u);
            let commit_gen_pool = self.commit_edge.map(|(g, _)| g);
            for (slot, seq) in slots.into_iter().zip(result.per_seq) {
                if sampler_split && seq.logits_handle.is_some() {
                    to_sample.push((slot, seq.req_id, seq.logits_handle, seq.locator.clone()));
                } else if und_pool == Some(pool_idx) && seq.logits_handle.is_none() {
                    // A locator on a publish-side result is the conditioning KV the
                    // gen pool must fetch (disambiguated from a sampler-handoff
                    // logits locator by the absent logits_handle).
                    if let Some(locator) = &seq.locator {
                        to_record.push((seq.req_id, locator.clone()));
                    }
                } else if commit_gen_pool == Some(pool_idx)
                    && seq.logits_handle.is_none()
                    && let Some(locator) = &seq.locator
                {
                    to_record_commit.push((seq.req_id, locator.clone()));
                }
                step.outputs[slot] = Some(seq);
            }
            step.expected &= !bit;
            step.samples_pending += to_sample.len() as u64;
            step.worker_exec_us = step
                .worker_exec_us
                .saturating_add(result.worker_exec_us.unwrap_or(0));
        }
        // Record the conditioning-KV locators so the matching `denoise_gen` op
        // (a later step, routed to the gen pool) is threaded with them.
        for (req_id, locator) in to_record {
            self.mover.record_conditioning(req_id, locator, step_id);
        }
        for (req_id, locator) in to_record_commit {
            self.mover.record_commit_latent(req_id, locator);
        }
        // Second phase: dispatch the synthesized `sample` ops to the Sampler pool.
        for (slot, req_id, logits_handle, locator) in to_sample {
            self.synthesize_sample(step_id, slot, req_id, logits_handle, locator)?;
        }
        self.try_complete(step_id)
    }

    /// Dispatch one synthesized `sample` op to the Sampler pool, carrying the
    /// logits handle/locator and the request's sampling params.
    fn synthesize_sample(
        &mut self,
        step_id: u64,
        slot: usize,
        req_id: RequestId,
        logits_handle: Option<u64>,
        locator: Option<String>,
    ) -> anyhow::Result<()> {
        let sampler = self
            .sampler_pool
            .ok_or_else(|| anyhow::anyhow!("synthesize_sample with no sampler pool"))?;
        let op_id = self.next_sample_op_id;
        self.next_sample_op_id = self.next_sample_op_id.wrapping_add(1);
        let new_reqs: Vec<NewRequestData> = self
            .req_sampling
            .get(&req_id)
            .cloned()
            .into_iter()
            .collect();
        let op = ForwardOp {
            req_id,
            kind: OpKind::Sample,
            logits_handle,
            locator,
            op_id: Some(op_id),
            ..Default::default()
        };
        self.pools[sampler].exec.submit(ForwardBatch {
            step_id,
            new_reqs,
            ops: vec![op],
        })?;
        self.pending_samples.insert(op_id, (step_id, slot));
        Ok(())
    }

    /// Backfill sampled tokens from a Sampler-pool result into the steps that
    /// synthesized them, correlating by the echoed op_id.
    fn backfill_samples(&mut self, result: ForwardResult) -> anyhow::Result<()> {
        let mut touched: Vec<u64> = Vec::new();
        for seq in result.per_seq {
            let Some(op_id) = seq.op_id else {
                anyhow::bail!("sampler result missing op_id for backfill");
            };
            let Some((step_id, slot)) = self.pending_samples.remove(&op_id) else {
                anyhow::bail!("sampler result op_id {op_id} matches no pending sample");
            };
            if let Some(step) = self.pending.get_mut(&step_id) {
                if let Some(Some(out)) = step.outputs.get_mut(slot) {
                    out.sampled_token_id = seq.sampled_token_id;
                    out.sampled_logprob = seq.sampled_logprob;
                    out.top_logprobs = seq.top_logprobs;
                }
                step.samples_pending = step.samples_pending.saturating_sub(1);
                step.worker_exec_us = step
                    .worker_exec_us
                    .saturating_add(result.worker_exec_us.unwrap_or(0));
                touched.push(step_id);
            }
        }
        for step_id in touched {
            self.try_complete(step_id)?;
        }
        Ok(())
    }

    fn try_complete(&mut self, step_id: u64) -> anyhow::Result<()> {
        let ready = self
            .pending
            .get(&step_id)
            .is_some_and(|step| step.expected == 0 && step.samples_pending == 0);
        if !ready {
            return Ok(());
        }
        let step = self
            .pending
            .remove(&step_id)
            .ok_or_else(|| anyhow::anyhow!("pending step {step_id} vanished during completion"))?;
        let mut merged = Vec::with_capacity(step.outputs.len());
        for slot in step.outputs {
            merged.push(slot.ok_or_else(|| {
                anyhow::anyhow!("stage merge missed a sequence result for step {step_id}")
            })?);
        }
        self.ready.push_back(ForwardResult {
            step_id,
            per_seq: merged,
            worker_exec_us: Some(step.worker_exec_us),
            forward_stats: None,
        });
        Ok(())
    }
}

impl Executor for StageRouter {
    fn caps(&self) -> EngineCaps {
        self.caps.clone()
    }

    fn pipeline_depth(&self) -> usize {
        self.depth
    }

    fn in_flight(&self) -> usize {
        self.pending.len() + self.ready.len()
    }

    fn generated_image_commit_capabilities(
        &self,
    ) -> uniserve_core::GeneratedImageCommitCapabilities {
        if self.commit_edge.is_some() {
            uniserve_core::GeneratedImageCommitCapabilities {
                inline: false,
                separate_writeback: true,
            }
        } else {
            uniserve_core::GeneratedImageCommitCapabilities {
                inline: true,
                separate_writeback: false,
            }
        }
    }

    fn can_submit(&self) -> bool {
        self.pending.len() < self.depth && self.pools.iter().all(|p| p.exec.can_submit())
    }

    fn submit(&mut self, batch: ForwardBatch) -> anyhow::Result<()> {
        self.pump()?;
        if batch.ops.is_empty() {
            return Ok(());
        }
        // Remember each request's admission descriptor so an edge that dispatches
        // a later op to a pool the request never reached at admission can replay
        // it: the decode→sampler `sample` op (Sampler pool) and the und→gen
        // `denoise_gen` op (gen pool, which needs the request's image params).
        if self.sampler_pool.is_some() || self.tower_edge.is_some() {
            for nr in &batch.new_reqs {
                self.req_sampling.insert(nr.req_id, nr.clone());
            }
        }
        let gen_pool = self.tower_edge.map(|(_, g)| g);
        let commit_und_pool = self.commit_edge.map(|(_, u)| u);
        let mut partitions: Vec<Vec<ForwardOp>> =
            (0..self.pools.len()).map(|_| Vec::new()).collect();
        let mut routes = Vec::with_capacity(batch.ops.len());
        // new_reqs to replay to the gen pool: the admission descriptor for each
        // request whose first denoise op we thread below.
        let mut gen_new_reqs: Vec<NewRequestData> = Vec::new();
        for op in &batch.ops {
            let idx = self.route_for(op)?;
            routes.push(idx);
            let mut op = op.clone();
            // Mode A: thread the conditioning-KV locator the und pool published
            // into this request's `denoise_gen` op so the gen pool fetches the
            // text KV it never produced (read-driven crossing, §4.2). Inert when
            // there is no und→gen edge or nothing was recorded.
            if Some(idx) == gen_pool
                && op.kind == OpKind::DenoiseGen
                && op.locator.is_none()
                && let Some(locator) = self.mover.take_conditioning(op.req_id)
            {
                op.locator = Some(locator);
                // First denoise step for this request (the step that consumes
                // the locator): replay its admission new_req so the gen worker
                // registers the request's image params it never saw.
                if let Some(nr) = self.req_sampling.get(&op.req_id) {
                    gen_new_reqs.push(nr.clone());
                }
            }
            if Some(idx) == commit_und_pool
                && op.kind == OpKind::CommitWriteback
                && op.locator.is_none()
                && let Some(locator) = self.mover.take_commit_latent(op.req_id)
            {
                op.locator = Some(locator);
            }
            partitions[idx].push(op);
        }
        let mut expected = 0u64;
        for (idx, ops) in partitions.into_iter().enumerate() {
            let new_reqs = if Some(idx) == gen_pool && !gen_new_reqs.is_empty() {
                gen_new_reqs.clone()
            } else {
                batch.new_reqs.clone()
            };
            if Self::submit_partition(&mut *self.pools[idx].exec, batch.step_id, new_reqs, ops)? {
                expected |= Self::pool_bit(idx);
            }
        }
        self.pending.insert(
            batch.step_id,
            PendingStep {
                expected,
                outputs: routes.iter().map(|_| None).collect(),
                routes,
                samples_pending: 0,
                worker_exec_us: 0,
            },
        );
        // A step with no submitted partition (e.g. all ops empty) is already
        // complete; flush it so the scheduler sees an immediate result.
        self.try_complete(batch.step_id)?;
        Ok(())
    }

    fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>> {
        self.pump()?;
        Ok(self.ready.pop_front())
    }

    fn check_liveness(&mut self) -> anyhow::Result<()> {
        for (idx, pool) in self.pools.iter_mut().enumerate() {
            pool.exec
                .check_liveness()
                .map_err(|e| anyhow::anyhow!("stage pool {idx} liveness: {e}"))?;
        }
        Ok(())
    }

    fn event_driven(&self) -> bool {
        // Event-driven only when every pool is, so the scheduler's single park
        // covers the whole topology; otherwise it polls (the safe default).
        !self.pools.is_empty() && self.pools.iter().all(|p| p.exec.event_driven())
    }

    fn command_waker(&self) -> CommandWaker {
        // Command ingress only needs to break the park; the first pool's waker
        // does that (the scheduler then drains every pool).
        self.pools
            .first()
            .map(|p| p.exec.command_waker())
            .unwrap_or_else(CommandWaker::noop)
    }

    fn park_for_event(&mut self, timeout: std::time::Duration) -> anyhow::Result<()> {
        // Park on a pool that currently has work in flight (its result wake).
        // Pools are independent, so a result arriving on a *different* pool is
        // surfaced when the scheduler re-drains all pools after this park returns
        // — within the park's safety-net slice in the worst case. Without a
        // cross-endpoint WaitSet this is the prompt-and-correct behavior.
        let idx = self
            .pools
            .iter()
            .position(|p| p.exec.in_flight() > 0)
            .unwrap_or(0);
        match self.pools.get_mut(idx) {
            Some(pool) => pool.exec.park_for_event(timeout),
            None => {
                std::thread::sleep(timeout.min(std::time::Duration::from_millis(1)));
                Ok(())
            }
        }
    }

    fn next_result(&mut self) -> anyhow::Result<ForwardResult> {
        loop {
            if let Some(result) = self.poll()? {
                return Ok(result);
            }
            if self.pending.is_empty() {
                anyhow::bail!("next_result called with no in-flight stage batches");
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    fn stage_ready(&self, req_id: RequestId) -> bool {
        self.mover.request_ready(req_id)
    }

    fn control(&mut self, op: ControlOp) -> anyhow::Result<u64> {
        let call_id = self.next_call_id;
        self.next_call_id += 1;
        for pool in self.control_targets(&op) {
            self.pools[pool].exec.control(op.clone())?;
        }
        Ok(call_id)
    }

    fn control_wait(
        &mut self,
        op: ControlOp,
        targets: Option<&[u32]>,
    ) -> anyhow::Result<Vec<ControlAck>> {
        // Pools are independent executors that each report ranks from 0; shift
        // each pool's ranks above the previous so merged acks never collide.
        let mut acks = Vec::new();
        let mut rank_offset = 0u32;
        for pool in self.control_targets(&op) {
            let pool_acks = self.pools[pool].exec.control_wait(op.clone(), targets)?;
            let next_offset = pool_acks
                .iter()
                .map(|ack| ack.rank.saturating_add(1))
                .max()
                .unwrap_or(0)
                .saturating_add(rank_offset);
            for mut ack in pool_acks {
                ack.rank = ack.rank.saturating_add(rank_offset);
                acks.push(ack);
            }
            rank_offset = next_offset;
        }
        Ok(acks)
    }

    fn shutdown(&mut self) {
        for pool in self.pools.iter_mut() {
            pool.exec.shutdown();
        }
    }
}

impl StageRouter {
    /// Which pools a control op fans out to. `FreeEncoder` only reaches
    /// encoder-capable pools (they hold the encoder cache); every other control
    /// broadcasts. Returns pool indices.
    fn control_targets(&self, op: &ControlOp) -> Vec<usize> {
        match op {
            ControlOp::FreeEncoder(_) => {
                let encoder_pools: Vec<usize> = self
                    .pools
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| {
                        p.kind.handles(OpKind::VitEncode) || p.kind.handles(OpKind::VaeEncode)
                    })
                    .map(|(idx, _)| idx)
                    .collect();
                if encoder_pools.is_empty() {
                    (0..self.pools.len()).collect()
                } else {
                    encoder_pools
                }
            }
            _ => (0..self.pools.len()).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use uniserve_core::{Modality, RequestId};
    use uniserve_worker_wire::{EngineCaps, ForwardOp, OpKind};

    /// Records submitted batches and echoes one SeqResult per op (req_id as the
    /// sampled token), so merges can be checked against the original op order.
    struct PoolExec {
        caps: EngineCaps,
        queued: VecDeque<ForwardResult>,
    }

    impl PoolExec {
        fn new() -> Self {
            let caps = EngineCaps {
                pipeline_depth: 4,
                ..Default::default()
            };
            Self {
                caps,
                queued: VecDeque::new(),
            }
        }

        fn for_kind(kind: WorkerKind) -> Self {
            let mut exec = Self::new();
            exec.caps.supported_ops = kind.supported_ops().to_vec();
            exec
        }
    }

    impl Executor for PoolExec {
        fn caps(&self) -> EngineCaps {
            self.caps.clone()
        }
        fn pipeline_depth(&self) -> usize {
            4
        }
        fn in_flight(&self) -> usize {
            self.queued.len()
        }
        fn submit(&mut self, batch: ForwardBatch) -> anyhow::Result<()> {
            let per_seq = batch
                .ops
                .iter()
                .map(|op| SeqResult {
                    req_id: op.req_id,
                    sampled_token_id: Some(op.req_id.0 as u32),
                    ..Default::default()
                })
                .collect();
            self.queued.push_back(ForwardResult {
                step_id: batch.step_id,
                per_seq,
                worker_exec_us: Some(10),
                forward_stats: None,
            });
            Ok(())
        }
        fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>> {
            Ok(self.queued.pop_front())
        }
        fn next_result(&mut self) -> anyhow::Result<ForwardResult> {
            self.queued
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("no queued result"))
        }
        fn control(&mut self, _op: ControlOp) -> anyhow::Result<u64> {
            Ok(1)
        }
        fn control_wait(
            &mut self,
            _op: ControlOp,
            _targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            Ok(vec![ControlAck {
                rank: 0,
                ok: true,
                message: None,
            }])
        }
    }

    #[test]
    fn staged_capabilities_follow_the_pool_that_executes_each_resource_op() {
        let mut und_caps = EngineCaps {
            supported_ops: vec![
                OpKind::PrefillUnd,
                OpKind::DecodeUnd,
                OpKind::TargetVerifyUnd,
                OpKind::CommitWriteback,
                OpKind::VaeEncode,
                OpKind::VitEncode,
                OpKind::Sample,
            ],
            max_latent_size: 0,
            latent_downsample: 0,
            max_vae_grid_tokens: 321,
            max_vit_grid_tokens: 654,
            scratch_capacity_tokens: 10_000,
            encoder_cache_budget: 12,
            ..EngineCaps::default()
        };
        und_caps.pipeline_depth = 5;
        let mut gen_caps = EngineCaps {
            supported_ops: vec![OpKind::DenoiseGen, OpKind::CommitGen, OpKind::EncodeFrame],
            max_latent_size: 4_096,
            latent_downsample: 16,
            max_cfg_branches: 2,
            scratch_capacity_tokens: 777,
            commit_marker_tokens: 3,
            gen_rope_advance: 4,
            ..EngineCaps::default()
        };
        gen_caps.pipeline_depth = 3;
        let router = StageRouter::try_new(vec![
            (
                WorkerKind::Und,
                Box::new(PoolExec {
                    caps: und_caps,
                    queued: VecDeque::new(),
                }),
            ),
            (
                WorkerKind::Gen,
                Box::new(PoolExec {
                    caps: gen_caps,
                    queued: VecDeque::new(),
                }),
            ),
        ])
        .expect("compatible staged pools");

        let caps = router.caps();
        assert_eq!(caps.max_latent_size, 4_096);
        assert_eq!(caps.latent_downsample, 16);
        assert_eq!(caps.max_cfg_branches, 2);
        assert_eq!(caps.scratch_capacity_tokens, 777);
        assert_eq!(caps.max_vae_grid_tokens, 321);
        assert_eq!(caps.max_vit_grid_tokens, 654);
        assert_eq!(caps.encoder_cache_budget, 12);
        assert_eq!(caps.commit_marker_tokens, 3);
        assert_eq!(caps.gen_rope_advance, 4);
        assert_eq!(caps.pipeline_depth, 3);
    }

    #[test]
    fn staged_kv_pools_reject_incompatible_block_geometry() {
        let prefill_caps = EngineCaps {
            block_size: 16,
            supported_ops: vec![OpKind::PrefillUnd],
            ..EngineCaps::default()
        };
        let decode_caps = EngineCaps {
            block_size: 32,
            supported_ops: vec![OpKind::DecodeUnd, OpKind::TargetVerifyUnd],
            ..EngineCaps::default()
        };

        let error = StageRouter::try_new(vec![
            (
                WorkerKind::Prefill,
                Box::new(PoolExec {
                    caps: prefill_caps,
                    queued: VecDeque::new(),
                }),
            ),
            (
                WorkerKind::Decode,
                Box::new(PoolExec {
                    caps: decode_caps,
                    queued: VecDeque::new(),
                }),
            ),
        ])
        .err()
        .expect("incompatible block sizes must fail");
        assert!(error.to_string().contains("block_size"));
    }

    fn op(req_id: u64, kind: OpKind, modality: Modality) -> ForwardOp {
        ForwardOp {
            req_id: RequestId(req_id),
            kind,
            modality,
            ..Default::default()
        }
    }

    /// Role-specific mock for the decode→sampler edge: a `decode` pool defers
    /// sampling (returns a logits handle + locator, no token); a `sampler` pool
    /// consumes a synthesized `sample` op (echoes its op_id + a token).
    struct RoleExec {
        role: &'static str,
        queued: VecDeque<ForwardResult>,
    }

    impl RoleExec {
        fn new(role: &'static str) -> Self {
            Self {
                role,
                queued: VecDeque::new(),
            }
        }
    }

    struct CommitEdgeExec {
        role: &'static str,
        queued: VecDeque<ForwardResult>,
    }

    impl CommitEdgeExec {
        fn new(role: &'static str) -> Self {
            Self {
                role,
                queued: VecDeque::new(),
            }
        }
    }

    impl Executor for CommitEdgeExec {
        fn caps(&self) -> EngineCaps {
            EngineCaps {
                pipeline_depth: 4,
                supported_ops: if self.role == "gen" {
                    vec![OpKind::DenoiseGen, OpKind::CommitGen, OpKind::EncodeFrame]
                } else {
                    vec![
                        OpKind::PrefillUnd,
                        OpKind::DecodeUnd,
                        OpKind::TargetVerifyUnd,
                        OpKind::CommitWriteback,
                        OpKind::VaeEncode,
                        OpKind::VitEncode,
                        OpKind::Sample,
                    ]
                },
                ..Default::default()
            }
        }
        fn pipeline_depth(&self) -> usize {
            4
        }
        fn in_flight(&self) -> usize {
            self.queued.len()
        }
        fn submit(&mut self, batch: ForwardBatch) -> anyhow::Result<()> {
            let per_seq = batch
                .ops
                .iter()
                .map(|op| {
                    if self.role == "gen" && op.kind == OpKind::CommitGen {
                        SeqResult {
                            req_id: op.req_id,
                            locator: Some("latent-locator".into()),
                            image_png_b64: Some("png".into()),
                            ..Default::default()
                        }
                    } else if self.role == "und" && op.kind == OpKind::CommitWriteback {
                        SeqResult {
                            req_id: op.req_id,
                            locator: op.locator.clone(),
                            sampled_token_id: Some(1234),
                            ..Default::default()
                        }
                    } else {
                        SeqResult {
                            req_id: op.req_id,
                            ..Default::default()
                        }
                    }
                })
                .collect();
            self.queued.push_back(ForwardResult {
                step_id: batch.step_id,
                per_seq,
                worker_exec_us: Some(10),
                forward_stats: None,
            });
            Ok(())
        }
        fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>> {
            Ok(self.queued.pop_front())
        }
        fn next_result(&mut self) -> anyhow::Result<ForwardResult> {
            self.queued
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("no queued result"))
        }
        fn control(&mut self, _op: ControlOp) -> anyhow::Result<u64> {
            Ok(1)
        }
        fn control_wait(
            &mut self,
            _op: ControlOp,
            _targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            Ok(vec![ControlAck {
                rank: 0,
                ok: true,
                message: None,
            }])
        }
    }

    impl Executor for RoleExec {
        fn caps(&self) -> EngineCaps {
            EngineCaps {
                pipeline_depth: 4,
                supported_ops: if self.role == "sampler" {
                    vec![OpKind::Sample]
                } else {
                    vec![
                        OpKind::DecodeUnd,
                        OpKind::TargetVerifyUnd,
                        OpKind::DenoiseGen,
                        OpKind::CommitGen,
                        OpKind::CommitWriteback,
                    ]
                },
                ..Default::default()
            }
        }
        fn pipeline_depth(&self) -> usize {
            4
        }
        fn in_flight(&self) -> usize {
            self.queued.len()
        }
        fn submit(&mut self, batch: ForwardBatch) -> anyhow::Result<()> {
            let role = self.role;
            let per_seq = batch
                .ops
                .iter()
                .map(|op| {
                    if role == "sampler" {
                        // Consume a synthesized sample op: echo op_id + a token.
                        SeqResult {
                            req_id: op.req_id,
                            sampled_token_id: Some(op.req_id.0 as u32 + 1000),
                            op_id: op.op_id,
                            ..Default::default()
                        }
                    } else {
                        // Decode defers sampling: publish logits, return handle+locator.
                        SeqResult {
                            req_id: op.req_id,
                            logits_handle: Some(op.req_id.0 * 10),
                            locator: Some("ZGVjb2RlLWxvY2F0b3I=".into()),
                            op_id: op.op_id,
                            ..Default::default()
                        }
                    }
                })
                .collect();
            self.queued.push_back(ForwardResult {
                step_id: batch.step_id,
                per_seq,
                worker_exec_us: Some(5),
                forward_stats: None,
            });
            Ok(())
        }
        fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>> {
            Ok(self.queued.pop_front())
        }
        fn next_result(&mut self) -> anyhow::Result<ForwardResult> {
            self.queued
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("no queued result"))
        }
        fn control(&mut self, _op: ControlOp) -> anyhow::Result<u64> {
            Ok(1)
        }
        fn control_wait(
            &mut self,
            _op: ControlOp,
            _targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            Ok(vec![ControlAck {
                rank: 0,
                ok: true,
                message: None,
            }])
        }
    }

    #[test]
    fn decode_sampler_two_phase_backfills_token() {
        // Decode pool + Sampler pool: a decode op that defers sampling must
        // produce a synthesized `sample` op to the Sampler, whose token is
        // backfilled into the decode result before the step finalizes.
        let mut router = StageRouter::new(vec![
            (WorkerKind::Decode, Box::new(RoleExec::new("decode"))),
            (WorkerKind::Sampler, Box::new(RoleExec::new("sampler"))),
        ]);
        router
            .submit(ForwardBatch {
                step_id: 9,
                new_reqs: vec![NewRequestData::new(RequestId(7))],
                ops: vec![op(7, OpKind::DecodeUnd, Modality::Und)],
            })
            .unwrap();
        let mut out = None;
        for _ in 0..10 {
            if let Some(r) = router.poll().unwrap() {
                out = Some(r);
                break;
            }
        }
        let out = out.expect("decode→sampler merged result");
        assert_eq!(out.step_id, 9);
        assert_eq!(out.per_seq.len(), 1);
        assert_eq!(out.per_seq[0].req_id, RequestId(7));
        // The token came from the Sampler pool (1000 + req_id), not the decode pool.
        assert_eq!(out.per_seq[0].sampled_token_id, Some(1007));
        // The decode op's logits handle is retained on the merged result.
        assert_eq!(out.per_seq[0].logits_handle, Some(70));
    }

    #[test]
    fn no_sampler_pool_leaves_decode_result_untouched() {
        // Without a Sampler pool, a decode result (even with a logits handle) is
        // finalized as-is — the 2-phase path is inert.
        let mut router = StageRouter::new(vec![(
            WorkerKind::Decode,
            Box::new(RoleExec::new("decode")),
        )]);
        router
            .submit(ForwardBatch {
                step_id: 3,
                new_reqs: Vec::new(),
                ops: vec![op(7, OpKind::DecodeUnd, Modality::Und)],
            })
            .unwrap();
        let out = router.poll().unwrap().expect("decode result");
        assert_eq!(out.per_seq[0].sampled_token_id, None);
        assert_eq!(out.per_seq[0].logits_handle, Some(70));
    }

    #[test]
    fn single_pool_is_pass_through_in_order() {
        let mut router = StageRouter::new(vec![(
            WorkerKind::Full,
            Box::new(uniserve_testkit::StubExecutor::new().with_pipeline_depth(4)),
        )]);
        router
            .submit(ForwardBatch {
                step_id: 5,
                new_reqs: Vec::new(),
                ops: vec![
                    op(10, OpKind::PrefillUnd, Modality::Und),
                    op(20, OpKind::DenoiseGen, Modality::Gen),
                    op(30, OpKind::DecodeUnd, Modality::Und),
                ],
            })
            .unwrap();
        let out = router.poll().unwrap().expect("merged result");
        assert_eq!(out.step_id, 5);
        let ids: Vec<u64> = out.per_seq.iter().map(|s| s.req_id.0).collect();
        assert_eq!(ids, vec![10, 20, 30]);
    }

    #[test]
    fn two_role_merges_in_original_op_order() {
        let mut router = StageRouter::new(vec![
            (WorkerKind::Und, Box::new(PoolExec::new())),
            (WorkerKind::Gen, Box::new(PoolExec::new())),
        ]);
        router
            .submit(ForwardBatch {
                step_id: 7,
                new_reqs: Vec::new(),
                ops: vec![
                    op(10, OpKind::DecodeUnd, Modality::Und),
                    op(20, OpKind::DenoiseGen, Modality::Gen),
                    op(30, OpKind::DecodeUnd, Modality::Und),
                ],
            })
            .unwrap();
        let out = router.poll().unwrap().expect("merged result");
        assert_eq!(out.step_id, 7);
        let ids: Vec<u64> = out.per_seq.iter().map(|s| s.req_id.0).collect();
        assert_eq!(ids, vec![10, 20, 30]);
        assert_eq!(out.worker_exec_us, Some(20));
    }

    #[test]
    fn three_pool_epd_routes_by_op_kind() {
        // Encoder + Prefill + Decode pools; ops fan to the owning pool and merge.
        let mut router = StageRouter::new(vec![
            (
                WorkerKind::Encoder,
                Box::new(PoolExec::for_kind(WorkerKind::Encoder)),
            ),
            (
                WorkerKind::Prefill,
                Box::new(PoolExec::for_kind(WorkerKind::Prefill)),
            ),
            (
                WorkerKind::Decode,
                Box::new(PoolExec::for_kind(WorkerKind::Decode)),
            ),
        ]);
        router
            .submit(ForwardBatch {
                step_id: 3,
                new_reqs: Vec::new(),
                ops: vec![
                    op(1, OpKind::VitEncode, Modality::Und),
                    op(2, OpKind::PrefillUnd, Modality::Und),
                    op(3, OpKind::DecodeUnd, Modality::Und),
                    op(4, OpKind::DenoiseGen, Modality::Gen),
                ],
            })
            .unwrap();
        let out = router.poll().unwrap().expect("merged result");
        let ids: Vec<u64> = out.per_seq.iter().map(|s| s.req_id.0).collect();
        assert_eq!(ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn und_gen_pools_route_understanding_and_generation_by_op_kind() {
        // The production und/gen path: StageRouter::new over Und/Gen pools (what
        // `--workers und:1,gen:1` composes) routes understanding ops to the und
        // pool and generation ops to the gen pool, merging in original order.
        let mut router = StageRouter::new(vec![
            (
                WorkerKind::Und,
                Box::new(PoolExec::for_kind(WorkerKind::Und)),
            ),
            (
                WorkerKind::Gen,
                Box::new(PoolExec::for_kind(WorkerKind::Gen)),
            ),
        ]);
        assert_eq!(
            router.generated_image_commit_capabilities(),
            uniserve_core::GeneratedImageCommitCapabilities {
                inline: false,
                separate_writeback: true,
            }
        );
        router
            .submit(ForwardBatch {
                step_id: 11,
                new_reqs: Vec::new(),
                ops: vec![
                    op(1, OpKind::PrefillUnd, Modality::Und),
                    op(2, OpKind::DenoiseGen, Modality::Gen),
                    op(3, OpKind::DecodeUnd, Modality::Und),
                    op(4, OpKind::CommitGen, Modality::Gen),
                    op(5, OpKind::VitEncode, Modality::Und),
                ],
            })
            .unwrap();
        let out = router.poll().unwrap().expect("merged und/gen result");
        let ids: Vec<u64> = out.per_seq.iter().map(|s| s.req_id.0).collect();
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn commit_writeback_receives_commit_gen_latent_locator() {
        let mut router = StageRouter::new(vec![
            (WorkerKind::Und, Box::new(CommitEdgeExec::new("und"))),
            (WorkerKind::Gen, Box::new(CommitEdgeExec::new("gen"))),
        ]);
        router
            .submit(ForwardBatch {
                step_id: 21,
                new_reqs: Vec::new(),
                ops: vec![op(9, OpKind::CommitGen, Modality::Gen)],
            })
            .unwrap();
        let out = router.poll().unwrap().expect("commit_gen result");
        assert_eq!(out.per_seq[0].locator.as_deref(), Some("latent-locator"));

        router
            .submit(ForwardBatch {
                step_id: 22,
                new_reqs: Vec::new(),
                ops: vec![op(9, OpKind::CommitWriteback, Modality::Und)],
            })
            .unwrap();
        let out = router.poll().unwrap().expect("commit_writeback result");
        assert_eq!(out.per_seq[0].locator.as_deref(), Some("latent-locator"));
        assert_eq!(out.per_seq[0].sampled_token_id, Some(1234));
    }

    #[test]
    fn control_wait_acks_have_distinct_ranks() {
        let mut router = StageRouter::new(vec![
            (WorkerKind::Und, Box::new(PoolExec::new())),
            (WorkerKind::Gen, Box::new(PoolExec::new())),
        ]);
        let acks = router
            .control_wait(ControlOp::ResetPrefixCache, None)
            .unwrap();
        assert_eq!(acks.len(), 2);
        assert_eq!(acks.iter().map(|a| a.rank).collect::<Vec<_>>(), vec![0, 1]);
    }

    /// Records every op it is submitted (shared handle) and emits one result per
    /// op, optionally stamping a conditioning `locator` (the und publish side).
    struct RecordingExec {
        recorded: std::sync::Arc<std::sync::Mutex<Vec<ForwardOp>>>,
        emit_locator: Option<String>,
        queued: VecDeque<ForwardResult>,
    }

    impl RecordingExec {
        fn new(
            recorded: std::sync::Arc<std::sync::Mutex<Vec<ForwardOp>>>,
            emit_locator: Option<String>,
        ) -> Self {
            Self {
                recorded,
                emit_locator,
                queued: VecDeque::new(),
            }
        }
    }

    impl Executor for RecordingExec {
        fn caps(&self) -> EngineCaps {
            EngineCaps {
                pipeline_depth: 4,
                ..Default::default()
            }
        }
        fn pipeline_depth(&self) -> usize {
            4
        }
        fn in_flight(&self) -> usize {
            self.queued.len()
        }
        fn submit(&mut self, batch: ForwardBatch) -> anyhow::Result<()> {
            let per_seq = batch
                .ops
                .iter()
                .map(|op| SeqResult {
                    req_id: op.req_id,
                    sampled_token_id: Some(op.req_id.0 as u32),
                    locator: self.emit_locator.clone(),
                    ..Default::default()
                })
                .collect();
            self.recorded
                .lock()
                .unwrap()
                .extend(batch.ops.iter().cloned());
            self.queued.push_back(ForwardResult {
                step_id: batch.step_id,
                per_seq,
                worker_exec_us: Some(10),
                forward_stats: None,
            });
            Ok(())
        }
        fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>> {
            Ok(self.queued.pop_front())
        }
        fn next_result(&mut self) -> anyhow::Result<ForwardResult> {
            self.queued
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("no queued result"))
        }
        fn control(&mut self, _op: ControlOp) -> anyhow::Result<u64> {
            Ok(1)
        }
        fn control_wait(
            &mut self,
            _op: ControlOp,
            _targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            Ok(vec![ControlAck {
                rank: 0,
                ok: true,
                message: None,
            }])
        }
    }

    #[test]
    fn tower_edge_threads_conditioning_locator_into_denoise() {
        // Mode A und→gen: the und pool publishes a conditioning locator on its
        // decode result; a later `denoise_gen` op for the same request, routed to
        // the gen pool, is threaded with that locator so the gen pool can fetch
        // the text KV it never produced.
        let und_ops = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let gen_ops = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let und = RecordingExec::new(std::sync::Arc::clone(&und_ops), Some("cond-7".to_string()));
        let gen_exec = RecordingExec::new(std::sync::Arc::clone(&gen_ops), None);
        let mut router = StageRouter::new(vec![
            (WorkerKind::Und, Box::new(und)),
            (WorkerKind::Gen, Box::new(gen_exec)),
        ]);

        // Step 1: und decode for req 7 → its result carries the conditioning locator.
        router
            .submit(ForwardBatch {
                step_id: 1,
                new_reqs: vec![],
                ops: vec![op(7, OpKind::DecodeUnd, Modality::Und)],
            })
            .unwrap();
        while router.poll().unwrap().is_some() {}

        // Step 2: denoise_gen for req 7 → routed to the gen pool, threaded.
        router
            .submit(ForwardBatch {
                step_id: 2,
                new_reqs: vec![],
                ops: vec![op(7, OpKind::DenoiseGen, Modality::Gen)],
            })
            .unwrap();
        while router.poll().unwrap().is_some() {}

        let gen_seen = gen_ops.lock().unwrap();
        let denoise = gen_seen
            .iter()
            .find(|o| o.kind == OpKind::DenoiseGen && o.req_id == RequestId(7))
            .expect("gen pool received the denoise op");
        assert_eq!(
            denoise.locator.as_deref(),
            Some("cond-7"),
            "the und-published conditioning locator must be threaded into the gen denoise op"
        );
    }

    #[test]
    fn tensor_mover_defaults_to_ready_with_no_edge() {
        // The non-disaggregated path records no edges, so every request is ready
        // and nothing is threaded (preserves the Phase-0 no-op behavior).
        let mut mover = TensorMover::new();
        assert!(mover.request_ready(RequestId(7)));
        assert_eq!(mover.take_conditioning(RequestId(7)), None);
    }

    #[test]
    fn tensor_mover_records_and_threads_conditioning_read_driven() {
        let mut mover = TensorMover::new();
        mover.record_conditioning(RequestId(7), "loc-7".to_string(), 1);
        // Read-driven: recording never gates — the und (producer) step must finish
        // and the gen (consumer) op must stay dispatchable to consume the locator.
        assert!(mover.request_ready(RequestId(7)));
        assert!(mover.request_ready(RequestId(8)));
        // Threading the locator into the denoise op hands it off exactly once.
        assert_eq!(
            mover.take_conditioning(RequestId(7)).as_deref(),
            Some("loc-7")
        );
        assert_eq!(mover.take_conditioning(RequestId(7)), None);
    }
}
