//! Typed control-plane routing across heterogeneous worker pools.
//!
//! A staged topology partitions each execution batch by exact
//! [`OperationType`], sends every session admission to a pool before that pool's
//! first operation for the session, and restores the scheduler's original
//! operation order when pool results complete. Tensor payloads remain in the
//! worker data plane; cross-pool references use typed published products.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::time::Duration;

use uniserve_core::{CommandWaker, RequestId};
use uniserve_executor::{ControlAck, ControlOp, Executor, WorkerKind};
use uniserve_worker_wire::{
    Admission, Batch, EngineCaps, ExecutionResult, KvLeaseDelta, Operation, OperationEnvelope,
    OperationResult, OperationType, ResultDelta, SequenceInput, SequenceMode, SequenceOperation,
    SessionProjection, WorkerForwardStats,
};

struct PoolEntry {
    exec: Box<dyn Executor>,
}

struct PendingStep {
    expected_pools: u64,
    routes: Vec<usize>,
    operations: Vec<OperationEnvelope>,
    outputs: Vec<Option<OperationResult>>,
    samples_pending: usize,
    worker_exec_us: u64,
    forward_stats: Option<WorkerForwardStats>,
}

struct PendingSample {
    step_id: u64,
    slot: usize,
    operation: OperationEnvelope,
    source_pool: usize,
    product_handle: u64,
}

#[derive(Clone)]
struct ProjectionState {
    epoch: u64,
    version: u64,
    last_op_id: u64,
    admission_digest: String,
    source_digest: String,
    last_sampled_token: Option<u32>,
}

/// Routes a canonical typed batch across the pools of a staged topology.
pub struct StageRouter {
    routing: HashMap<OperationType, usize>,
    pools: Vec<PoolEntry>,
    caps: EngineCaps,
    depth: usize,
    pending: BTreeMap<u64, PendingStep>,
    ready: VecDeque<ExecutionResult>,
    admissions: HashMap<RequestId, Admission>,
    admitted_pools: HashSet<(usize, RequestId)>,
    pending_samples: HashMap<(RequestId, u64, u64), PendingSample>,
    projections: HashMap<RequestId, ProjectionState>,
    sampler_pool: Option<usize>,
    separate_image_writeback: bool,
    next_call_id: u64,
}

fn extend_unique<T: Clone + Eq + std::hash::Hash>(target: &mut Vec<T>, incoming: Vec<T>) {
    let mut seen: HashSet<T> = target.iter().cloned().collect();
    for item in incoming {
        if seen.insert(item.clone()) {
            target.push(item);
        }
    }
}

impl StageRouter {
    /// Build a staged router after validating exact, non-overlapping pool claims.
    pub fn try_new(pools: Vec<(WorkerKind, Box<dyn Executor>)>) -> anyhow::Result<Self> {
        anyhow::ensure!(!pools.is_empty(), "StageRouter needs at least one pool");
        anyhow::ensure!(
            pools.len() <= u64::BITS as usize,
            "StageRouter supports at most {} pools",
            u64::BITS
        );

        let mut routing = HashMap::new();
        for (index, (kind, executor)) in pools.iter().enumerate() {
            let caps = executor.caps();
            for operation_type in kind.supported_operation_types() {
                if !caps.supported_operation_types.contains(operation_type) {
                    continue;
                }
                anyhow::ensure!(
                    routing.insert(*operation_type, index).is_none(),
                    "multiple staged pools claim operation type {operation_type}; replication requires an explicit replica-group executor"
                );
            }
        }
        anyhow::ensure!(
            !routing.is_empty(),
            "staged pools expose no routable operation types"
        );

        let pools: Vec<_> = pools
            .into_iter()
            .map(|(_, exec)| PoolEntry { exec })
            .collect();
        let caps = Self::merge_caps(&pools, &routing)?;
        let depth = pools
            .iter()
            .map(|pool| pool.exec.pipeline_depth())
            .min()
            .unwrap_or(1)
            .max(1);
        let sampler_pool = routing.get(&OperationType::SequenceSample).copied();
        let separate_image_writeback = matches!(
            (
                routing.get(&OperationType::MaterializeImage),
                routing.get(&OperationType::TransferKv),
            ),
            (Some(materialize), Some(transfer)) if materialize != transfer
        );

        Ok(Self {
            routing,
            pools,
            caps,
            depth,
            pending: BTreeMap::new(),
            ready: VecDeque::new(),
            admissions: HashMap::new(),
            admitted_pools: HashSet::new(),
            pending_samples: HashMap::new(),
            projections: HashMap::new(),
            sampler_pool,
            separate_image_writeback,
            next_call_id: 1,
        })
    }

    fn merge_caps(
        pools: &[PoolEntry],
        routing: &HashMap<OperationType, usize>,
    ) -> anyhow::Result<EngineCaps> {
        let routed_caps = |operation_type: OperationType| {
            routing
                .get(&operation_type)
                .map(|index| pools[*index].exec.caps())
        };
        let mut kv_pool_indices = [
            OperationType::SequenceExtend,
            OperationType::SequenceDecode,
            OperationType::SequenceVerify,
            OperationType::MaterializeImage,
            OperationType::TransferKv,
        ]
        .into_iter()
        .filter_map(|operation_type| routing.get(&operation_type).copied())
        .collect::<Vec<_>>();
        kv_pool_indices.sort_unstable();
        kv_pool_indices.dedup();

        let seed_index = kv_pool_indices.first().copied().unwrap_or(0);
        let mut merged = pools[seed_index].exec.caps();
        let identities = pools
            .iter()
            .map(|pool| pool.exec.caps())
            .filter(|caps| !caps.model_spec_digest.is_empty() || !caps.weight_digest.is_empty())
            .map(|caps| (caps.model_spec_digest, caps.weight_digest))
            .collect::<Vec<_>>();
        if let Some(identity) = identities.first() {
            anyhow::ensure!(
                !identity.0.is_empty() && !identity.1.is_empty(),
                "model worker capability identity is incomplete"
            );
            anyhow::ensure!(
                identities.iter().all(|candidate| candidate == identity),
                "staged model workers expose different model or weight identities"
            );
            merged.model_spec_digest = identity.0.clone();
            merged.weight_digest = identity.1.clone();
        }

        if let Some(first_index) = kv_pool_indices.first().copied() {
            let first = pools[first_index].exec.caps();
            for index in kv_pool_indices.iter().copied().skip(1) {
                let other = pools[index].exec.caps();
                anyhow::ensure!(
                    other.block_size == first.block_size,
                    "staged KV pools disagree on block size"
                );
                anyhow::ensure!(
                    other.num_layers == first.num_layers
                        && other.kv_dtype == first.kv_dtype
                        && other.quantization == first.quantization
                        && other.groups == first.groups,
                    "staged KV pools expose incompatible cache layouts"
                );
            }
            merged.block_size = first.block_size;
            merged.num_blocks = kv_pool_indices
                .iter()
                .map(|index| pools[*index].exec.caps().num_blocks)
                .min()
                .unwrap_or(first.num_blocks);
            merged.num_layers = first.num_layers;
            merged.groups = first.groups;
            merged.kv_dtype = first.kv_dtype;
            merged.quantization = first.quantization;
            merged.bytes_per_token = kv_pool_indices
                .iter()
                .map(|index| pools[*index].exec.caps().bytes_per_token)
                .max()
                .unwrap_or(first.bytes_per_token);
        }

        merged.supported_operation_types = OperationType::ALL
            .into_iter()
            .filter(|operation_type| routing.contains_key(operation_type))
            .collect();
        merged.supported_controls.clear();
        merged.resource_classes.clear();
        for pool in pools {
            let caps = pool.exec.caps();
            extend_unique(&mut merged.supported_controls, caps.supported_controls);
            extend_unique(&mut merged.resource_classes, caps.resource_classes);
        }
        merged.pipeline_depth = pools
            .iter()
            .map(|pool| pool.exec.caps().pipeline_depth.max(1))
            .min()
            .unwrap_or(1);
        merged.execution_constraints.max_batch_operations = pools
            .iter()
            .map(|pool| pool.exec.caps().execution_constraints.max_batch_operations)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);

        let flow = routed_caps(OperationType::Flow);
        merged.max_latent_size = flow.as_ref().map_or(0, |caps| caps.max_latent_size);
        merged.latent_downsample = flow.as_ref().map_or(0, |caps| caps.latent_downsample);
        merged.max_cfg_branches = flow.as_ref().map_or(0, |caps| caps.max_cfg_branches);
        merged.scratch_capacity_tokens =
            flow.as_ref().map_or(0, |caps| caps.scratch_capacity_tokens);
        merged.max_vae_grid_tokens = routed_caps(OperationType::EncodeLatent)
            .as_ref()
            .map_or(0, |caps| caps.max_vae_grid_tokens);
        merged.max_vit_grid_tokens = routed_caps(OperationType::EncodeVision)
            .as_ref()
            .map_or(0, |caps| caps.max_vit_grid_tokens);
        if let Some(materialize) = routed_caps(OperationType::MaterializeImage) {
            merged.commit_marker_tokens = materialize.commit_marker_tokens;
            merged.gen_rope_advance = materialize.gen_rope_advance;
        }
        merged.encoder_cache_budget = [OperationType::EncodeLatent, OperationType::EncodeVision]
            .into_iter()
            .filter_map(|operation_type| {
                routed_caps(operation_type).map(|caps| caps.encoder_cache_budget)
            })
            .min()
            .unwrap_or(0);
        if let Some(decode) = routed_caps(OperationType::SequenceDecode) {
            merged.adapter_mode = decode.adapter_mode;
        }
        Ok(merged)
    }

    fn pool_bit(index: usize) -> u64 {
        1u64 << index
    }

    fn cache_admissions(&mut self, admissions: &[Admission]) -> anyhow::Result<()> {
        for admission in admissions {
            if let Some(existing) = self.admissions.get(&admission.session_id) {
                anyhow::ensure!(
                    existing == admission,
                    "session {} was readmitted with a different descriptor before drop",
                    admission.session_id.0
                );
            } else {
                self.admissions
                    .insert(admission.session_id, admission.clone());
            }
        }
        Ok(())
    }

    fn admissions_for(
        &self,
        pool_index: usize,
        operations: &[OperationEnvelope],
    ) -> anyhow::Result<Vec<Admission>> {
        let mut admissions = Vec::new();
        for operation in operations {
            if self
                .admitted_pools
                .contains(&(pool_index, operation.session_id))
            {
                continue;
            }
            let admission = self.admissions.get(&operation.session_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "pool {} has no admission for session {}",
                    pool_index,
                    operation.session_id.0
                )
            })?;
            anyhow::ensure!(
                admission.digest == operation.admission_digest,
                "operation admission digest changed for session {}",
                operation.session_id.0
            );
            admissions.push(admission.clone());
        }
        Ok(admissions)
    }

    fn projections_for(
        &self,
        operations: &[OperationEnvelope],
    ) -> anyhow::Result<Vec<SessionProjection>> {
        operations
            .iter()
            .filter(|operation| operation.base_version > 0)
            .map(|operation| {
                let state = self.projections.get(&operation.session_id).ok_or_else(|| {
                    anyhow::anyhow!(
                        "session {} has no committed projection for base version {}",
                        operation.session_id.0,
                        operation.base_version
                    )
                })?;
                anyhow::ensure!(
                    state.epoch == operation.epoch && state.version == operation.base_version,
                    "session {} projection is not at the operation base version",
                    operation.session_id.0
                );
                let projection = SessionProjection {
                    session_id: operation.session_id,
                    epoch: state.epoch,
                    version: state.version,
                    last_op_id: state.last_op_id,
                    admission_digest: state.admission_digest.clone(),
                    source_digest: state.source_digest.clone(),
                    last_sampled_token: state.last_sampled_token,
                };
                projection.validate_for(operation)?;
                Ok(projection)
            })
            .collect()
    }

    fn submit_partition(
        &mut self,
        pool_index: usize,
        step_id: u64,
        protocol_version: u16,
        operations: Vec<OperationEnvelope>,
    ) -> anyhow::Result<bool> {
        if operations.is_empty() {
            return Ok(false);
        }
        let admissions = self.admissions_for(pool_index, &operations)?;
        let projections = self.projections_for(&operations)?;
        let sessions = admissions
            .iter()
            .map(|admission| admission.session_id)
            .collect::<Vec<_>>();
        let partition = Batch {
            protocol_version,
            step_id,
            admissions,
            projections,
            operations,
        };
        partition.validate()?;
        self.pools[pool_index].exec.submit(partition)?;
        self.admitted_pools.extend(
            sessions
                .into_iter()
                .map(|session_id| (pool_index, session_id)),
        );
        Ok(true)
    }

    fn pump(&mut self) -> anyhow::Result<()> {
        for pool_index in 0..self.pools.len() {
            loop {
                let result = self.pools[pool_index].exec.poll()?;
                let Some(result) = result else {
                    break;
                };
                self.route_result(pool_index, result)?;
            }
        }
        Ok(())
    }

    fn route_result(&mut self, pool_index: usize, result: ExecutionResult) -> anyhow::Result<()> {
        if self.route_sample_result(pool_index, &result)? {
            return Ok(());
        }

        let step_id = result.step_id;
        let pool_bit = Self::pool_bit(pool_index);
        let mut samples = Vec::new();
        {
            let step = self
                .pending
                .get_mut(&step_id)
                .ok_or_else(|| anyhow::anyhow!("stage pool returned unknown step {step_id}"))?;
            anyhow::ensure!(
                step.expected_pools & pool_bit != 0,
                "pool {pool_index} returned a duplicate or unexpected result for step {step_id}"
            );
            let slots = step
                .routes
                .iter()
                .enumerate()
                .filter_map(|(slot, route)| (*route == pool_index).then_some(slot))
                .collect::<Vec<_>>();
            anyhow::ensure!(
                slots.len() == result.operations.len(),
                "pool {pool_index} returned {} operations for step {step_id}, expected {}",
                result.operations.len(),
                slots.len()
            );
            for (slot, operation_result) in slots.into_iter().zip(result.operations) {
                anyhow::ensure!(
                    step.outputs[slot].is_none(),
                    "pool {pool_index} returned operation slot {slot} twice for step {step_id}"
                );
                operation_result.validate_for(&step.operations[slot])?;
                if let Some(product) = published_logits(&operation_result) {
                    samples.push((slot, step.operations[slot].clone(), product));
                }
                step.outputs[slot] = Some(operation_result);
            }
            step.expected_pools &= !pool_bit;
            step.samples_pending += samples.len();
            step.worker_exec_us = step
                .worker_exec_us
                .saturating_add(result.worker_exec_us.unwrap_or(0));
            if step.forward_stats.is_none() {
                step.forward_stats = result.forward_stats;
            }
        }
        for (slot, operation, product) in samples {
            self.submit_sample(pool_index, step_id, slot, operation, product)?;
        }
        self.try_complete(step_id)
    }

    fn submit_sample(
        &mut self,
        source_pool: usize,
        step_id: u64,
        slot: usize,
        source: OperationEnvelope,
        product: uniserve_worker_wire::PublishedProduct,
    ) -> anyhow::Result<()> {
        let pool_index = self.sampler_pool.ok_or_else(|| {
            anyhow::anyhow!(
                "session {} published logits but the topology has no sequence sampler",
                source.session_id.0
            )
        })?;
        let Operation::Sequence(sequence) = &source.operation else {
            anyhow::bail!("only a sequence operation may publish logits");
        };
        anyhow::ensure!(
            matches!(sequence.mode, SequenceMode::Decode | SequenceMode::Verify),
            "operation type {} cannot publish logits for deferred sampling",
            source.operation_type()
        );
        let product_handle = product.handle;
        let mut operation = OperationEnvelope::unsealed(
            source.session_id,
            Operation::Sequence(SequenceOperation {
                mode: SequenceMode::Sample,
                lease: KvLeaseDelta::default(),
                position: sequence.position,
                policy: sequence.policy.clone(),
                input: SequenceInput::PublishedLogits(product),
            }),
        );
        operation.admission_digest = source.admission_digest.clone();
        operation.model_spec_digest = source.model_spec_digest.clone();
        operation.weight_digest = source.weight_digest.clone();
        operation.seal(source.epoch, source.op_id, source.base_version);
        let key = (source.session_id, source.epoch, source.op_id);
        anyhow::ensure!(
            !self.pending_samples.contains_key(&key),
            "duplicate deferred sample for session {} operation {}",
            source.session_id.0,
            source.op_id
        );
        self.pending_samples.insert(
            key,
            PendingSample {
                step_id,
                slot,
                operation: operation.clone(),
                source_pool,
                product_handle,
            },
        );
        if let Err(error) = self.submit_partition(
            pool_index,
            step_id,
            uniserve_worker_wire::EXECUTION_PROTOCOL_VERSION,
            vec![operation],
        ) {
            self.pending_samples.remove(&key);
            return Err(error);
        }
        Ok(())
    }

    fn route_sample_result(
        &mut self,
        pool_index: usize,
        result: &ExecutionResult,
    ) -> anyhow::Result<bool> {
        if self.sampler_pool != Some(pool_index) || result.operations.len() != 1 {
            return Ok(false);
        }
        let sample_result = &result.operations[0];
        let key = (
            sample_result.session_id,
            sample_result.epoch,
            sample_result.op_id,
        );
        let Some(sample) = self.pending_samples.remove(&key) else {
            return Ok(false);
        };
        anyhow::ensure!(
            result.step_id == sample.step_id,
            "sampler result step does not match its source step"
        );
        sample_result.validate_for(&sample.operation)?;
        let ResultDelta::Sequence(sample_delta) = &sample_result.delta else {
            anyhow::bail!("sequence sampler returned a non-sequence delta");
        };
        anyhow::ensure!(
            sample_delta.effect.published_logits.is_none(),
            "sequence sampler recursively published logits"
        );
        let step = self.pending.get_mut(&sample.step_id).ok_or_else(|| {
            anyhow::anyhow!("source step {} vanished before sampling", sample.step_id)
        })?;
        let output = step.outputs[sample.slot].as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "source output slot {} vanished before sampling",
                sample.slot
            )
        })?;
        let ResultDelta::Sequence(source_delta) = &mut output.delta else {
            anyhow::bail!("deferred sample source is not a sequence delta");
        };
        source_delta.effect.sampled_token_ids = sample_delta.effect.sampled_token_ids.clone();
        source_delta.effect.sampled_logprob = sample_delta.effect.sampled_logprob;
        source_delta.effect.top_logprobs = sample_delta.effect.top_logprobs.clone();
        source_delta.effect.published_logits = None;
        step.samples_pending = step.samples_pending.saturating_sub(1);
        step.worker_exec_us = step
            .worker_exec_us
            .saturating_add(result.worker_exec_us.unwrap_or(0));
        if sample.product_handle > 0 {
            let acknowledgements = self.pools[sample.source_pool].exec.control_wait(
                ControlOp::ReleaseProducts(vec![sample.product_handle]),
                None,
            )?;
            anyhow::ensure!(
                acknowledgements
                    .iter()
                    .all(|acknowledgement| acknowledgement.ok),
                "source pool could not release consumed deferred logits"
            );
        }
        self.try_complete(sample.step_id)?;
        Ok(true)
    }

    fn try_complete(&mut self, step_id: u64) -> anyhow::Result<()> {
        let complete = self
            .pending
            .get(&step_id)
            .is_some_and(|step| step.expected_pools == 0 && step.samples_pending == 0);
        if !complete {
            return Ok(());
        }
        let step = self
            .pending
            .remove(&step_id)
            .ok_or_else(|| anyhow::anyhow!("pending step {step_id} disappeared"))?;
        let operations = step
            .outputs
            .into_iter()
            .enumerate()
            .map(|(slot, output)| {
                output.ok_or_else(|| {
                    anyhow::anyhow!("stage merge missed operation slot {slot} for step {step_id}")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        for (operation, result) in step.operations.iter().zip(&operations) {
            let previous = self.projections.get(&operation.session_id);
            if let Some(previous) = previous {
                anyhow::ensure!(
                    previous.epoch == operation.epoch && previous.version == operation.base_version,
                    "session {} committed result is not contiguous with its projection",
                    operation.session_id.0
                );
            } else {
                anyhow::ensure!(
                    operation.base_version == 0,
                    "session {} first committed result does not start at version zero",
                    operation.session_id.0
                );
            }
            let sampled = result_sequence_effect(result)
                .and_then(|effect| effect.sampled_token_ids.last().copied())
                .or_else(|| previous.and_then(|state| state.last_sampled_token));
            self.projections.insert(
                operation.session_id,
                ProjectionState {
                    epoch: operation.epoch,
                    version: result.result_version,
                    last_op_id: operation.op_id,
                    admission_digest: operation.admission_digest.clone(),
                    source_digest: operation.digest.clone(),
                    last_sampled_token: sampled,
                },
            );
        }
        self.ready.push_back(ExecutionResult {
            step_id,
            operations,
            worker_exec_us: Some(step.worker_exec_us),
            forward_stats: step.forward_stats,
        });
        Ok(())
    }

    fn control_targets(&self, operation: &ControlOp) -> Vec<usize> {
        if matches!(operation, ControlOp::DropSession(_)) {
            return (0..self.pools.len()).collect();
        }
        self.pools
            .iter()
            .enumerate()
            .filter(|(_, pool)| {
                pool.exec
                    .caps()
                    .supported_controls
                    .contains(&operation.request_kind())
            })
            .map(|(index, _)| index)
            .collect()
    }

    fn forget_session(&mut self, session_id: RequestId) {
        self.admissions.remove(&session_id);
        self.admitted_pools
            .retain(|(_, admitted)| *admitted != session_id);
        self.pending_samples
            .retain(|(sampled, _, _), _| *sampled != session_id);
        self.projections.remove(&session_id);
    }
}

fn result_sequence_effect(
    result: &OperationResult,
) -> Option<&uniserve_worker_wire::SequenceEffect> {
    match &result.delta {
        ResultDelta::Sequence(delta) => Some(&delta.effect),
        ResultDelta::Materialize(delta) => delta.sequence.as_ref(),
        ResultDelta::Transfer(delta) => delta.sequence.as_ref(),
        ResultDelta::Flow(_) | ResultDelta::Encode(_) => None,
    }
}

fn published_logits(result: &OperationResult) -> Option<uniserve_worker_wire::PublishedProduct> {
    match &result.delta {
        ResultDelta::Sequence(delta) => delta.effect.published_logits.clone(),
        _ => None,
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
        uniserve_core::GeneratedImageCommitCapabilities {
            inline: !self.separate_image_writeback,
            separate_writeback: self.separate_image_writeback,
        }
    }

    fn can_submit(&self) -> bool {
        self.pending.len() < self.depth && self.pools.iter().all(|pool| pool.exec.can_submit())
    }

    fn submit(&mut self, batch: Batch) -> anyhow::Result<()> {
        batch.validate()?;
        self.pump()?;
        anyhow::ensure!(
            !self.pending.contains_key(&batch.step_id),
            "stage router already has step {} in flight",
            batch.step_id
        );
        self.cache_admissions(&batch.admissions)?;

        let mut partitions = (0..self.pools.len())
            .map(|_| Vec::new())
            .collect::<Vec<Vec<OperationEnvelope>>>();
        let mut routes = Vec::with_capacity(batch.operations.len());
        for operation in &batch.operations {
            let operation_type = operation.operation_type();
            let pool_index = self.routing.get(&operation_type).copied().ok_or_else(|| {
                anyhow::anyhow!(
                    "StageRouter has no pool for operation type {} in session {}",
                    operation_type,
                    operation.session_id.0
                )
            })?;
            if let Operation::Flow(flow) = &operation.operation {
                let sequence_pool = self
                    .routing
                    .get(&OperationType::SequenceDecode)
                    .or_else(|| self.routing.get(&OperationType::SequenceExtend));
                if sequence_pool.is_some_and(|sequence_pool| *sequence_pool != pool_index) {
                    let conditioning = flow.conditioning.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "cross-pool flow for session {} has no published KV conditioning",
                            operation.session_id.0
                        )
                    })?;
                    anyhow::ensure!(
                        !conditioning.locators.is_empty(),
                        "cross-pool flow for session {} has no data-plane KV locators",
                        operation.session_id.0
                    );
                }
            }
            routes.push(pool_index);
            partitions[pool_index].push(operation.clone());
        }

        let mut expected_pools = 0;
        for (pool_index, operations) in partitions.into_iter().enumerate() {
            if self.submit_partition(
                pool_index,
                batch.step_id,
                batch.protocol_version,
                operations,
            )? {
                expected_pools |= Self::pool_bit(pool_index);
            }
        }
        let operation_count = batch.operations.len();
        self.pending.insert(
            batch.step_id,
            PendingStep {
                expected_pools,
                routes,
                operations: batch.operations,
                outputs: (0..operation_count).map(|_| None).collect(),
                samples_pending: 0,
                worker_exec_us: 0,
                forward_stats: None,
            },
        );
        Ok(())
    }

    fn poll(&mut self) -> anyhow::Result<Option<ExecutionResult>> {
        self.pump()?;
        Ok(self.ready.pop_front())
    }

    fn check_liveness(&mut self) -> anyhow::Result<()> {
        for (index, pool) in self.pools.iter_mut().enumerate() {
            pool.exec
                .check_liveness()
                .map_err(|error| anyhow::anyhow!("stage pool {index} liveness: {error}"))?;
        }
        Ok(())
    }

    fn event_driven(&self) -> bool {
        !self.pools.is_empty() && self.pools.iter().all(|pool| pool.exec.event_driven())
    }

    fn command_waker(&self) -> CommandWaker {
        self.pools
            .first()
            .map(|pool| pool.exec.command_waker())
            .unwrap_or_else(CommandWaker::noop)
    }

    fn park_for_event(&mut self, timeout: Duration) -> anyhow::Result<()> {
        let index = self
            .pools
            .iter()
            .position(|pool| pool.exec.in_flight() > 0)
            .unwrap_or(0);
        match self.pools.get_mut(index) {
            Some(pool) => pool.exec.park_for_event(timeout),
            None => Ok(()),
        }
    }

    fn next_result(&mut self) -> anyhow::Result<ExecutionResult> {
        loop {
            if let Some(result) = self.poll()? {
                return Ok(result);
            }
            anyhow::ensure!(
                !self.pending.is_empty(),
                "next_result called with no in-flight stage batches"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn control(&mut self, operation: ControlOp) -> anyhow::Result<u64> {
        let call_id = self.next_call_id;
        self.next_call_id = self.next_call_id.saturating_add(1);
        let targets = self.control_targets(&operation);
        anyhow::ensure!(
            !targets.is_empty(),
            "no stage pool accepts control {}",
            operation.method()
        );
        for pool_index in targets {
            self.pools[pool_index].exec.control(operation.clone())?;
        }
        if let ControlOp::DropSession(session_id) = operation {
            self.forget_session(session_id);
        }
        Ok(call_id)
    }

    fn control_wait(
        &mut self,
        operation: ControlOp,
        targets: Option<&[u32]>,
    ) -> anyhow::Result<Vec<ControlAck>> {
        let pool_indices = self.control_targets(&operation);
        anyhow::ensure!(
            !pool_indices.is_empty(),
            "no stage pool accepts control {}",
            operation.method()
        );
        let mut acknowledgements = Vec::new();
        let mut rank_offset = 0;
        for pool_index in pool_indices {
            let mut pool_acknowledgements = self.pools[pool_index]
                .exec
                .control_wait(operation.clone(), targets)?;
            let rank_count = pool_acknowledgements
                .iter()
                .map(|ack| ack.rank.saturating_add(1))
                .max()
                .unwrap_or(0);
            for acknowledgement in &mut pool_acknowledgements {
                acknowledgement.rank = acknowledgement.rank.saturating_add(rank_offset);
            }
            acknowledgements.extend(pool_acknowledgements);
            rank_offset = rank_offset.saturating_add(rank_count);
        }
        if let ControlOp::DropSession(session_id) = operation {
            self.forget_session(session_id);
        }
        Ok(acknowledgements)
    }

    fn shutdown(&mut self) {
        for pool in &mut self.pools {
            pool.exec.shutdown();
        }
    }
}
