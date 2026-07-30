//! Typed control-plane routing across heterogeneous worker pools.
//!
//! A staged topology partitions each execution batch by exact [`WorkVariant`],
//! sends every session admission to a pool before that pool's first operation
//! for the session, and restores the scheduler's original operation order when
//! pool completions arrive. A worker-local device product is directly reachable
//! only within its producing pool. Cross-pool movement requires a declared
//! `Transfer(Product)` whose destination reference belongs to the consumer.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::time::Duration;

use uniserve_core::{CommandWaker, RequestId};
use uniserve_executor::{ControlAck, ControlOp, Executor, WorkerKind};
use uniserve_worker_wire::{
    Admission, Batch, CompletionRecord, CompletionReport, Control, EngineCaps, Operation,
    ProductPayload, RegistrationAck, RequestKey, WorkVariant, WorkerForwardStats,
};

struct PoolEntry {
    exec: Box<dyn Executor>,
}

struct PendingStep {
    expected_pools: u64,
    routes: Vec<usize>,
    operations: Vec<Operation>,
    outputs: Vec<Option<CompletionRecord>>,
    products: Vec<ProductPayload>,
    registration_visible: bool,
    worker_exec_us: u64,
    forward_stats: Option<WorkerForwardStats>,
}

/// Routes a canonical typed batch across the pools of a staged topology.
pub struct StageRouter {
    routing: HashMap<WorkVariant, usize>,
    pools: Vec<PoolEntry>,
    caps: EngineCaps,
    depth: usize,
    pending: BTreeMap<u64, PendingStep>,
    ready: VecDeque<CompletionReport>,
    admissions: HashMap<RequestKey, Admission>,
    admitted_pools: HashSet<(usize, RequestKey)>,
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
            for variant in kind.supported_work() {
                if !caps.supported_work.contains(variant) {
                    continue;
                }
                anyhow::ensure!(
                    routing.insert(*variant, index).is_none(),
                    "multiple staged pools claim work variant {variant:?}; replication requires an explicit replica-group executor"
                );
            }
        }
        anyhow::ensure!(
            !routing.is_empty(),
            "staged pools expose no routable work variants"
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
        Ok(Self {
            routing,
            pools,
            caps,
            depth,
            pending: BTreeMap::new(),
            ready: VecDeque::new(),
            admissions: HashMap::new(),
            admitted_pools: HashSet::new(),
            next_call_id: 1,
        })
    }

    fn merge_caps(
        pools: &[PoolEntry],
        routing: &HashMap<WorkVariant, usize>,
    ) -> anyhow::Result<EngineCaps> {
        let routed_caps =
            |variant: WorkVariant| routing.get(&variant).map(|index| pools[*index].exec.caps());
        let mut kv_pool_indices = [
            WorkVariant::TokenExtend,
            WorkVariant::TokenDecode,
            WorkVariant::TokenVerify,
            WorkVariant::Materialize,
            WorkVariant::TransferKvPublish,
            WorkVariant::TransferKvInstall,
        ]
        .into_iter()
        .filter_map(|variant| routing.get(&variant).copied())
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

        merged.supported_work = WorkVariant::ALL
            .into_iter()
            .filter(|variant| routing.contains_key(variant))
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

        let flow = routed_caps(WorkVariant::GenFlow);
        merged.max_latent_size = flow.as_ref().map_or(0, |caps| caps.max_latent_size);
        merged.latent_downsample = flow.as_ref().map_or(0, |caps| caps.latent_downsample);
        merged.max_cfg_branches = flow.as_ref().map_or(0, |caps| caps.max_cfg_branches);
        merged.scratch_capacity_tokens =
            flow.as_ref().map_or(0, |caps| caps.scratch_capacity_tokens);
        merged.max_vae_grid_tokens = routed_caps(WorkVariant::EncodeLatent)
            .as_ref()
            .map_or(0, |caps| caps.max_vae_grid_tokens);
        merged.max_vit_grid_tokens = routed_caps(WorkVariant::EncodeVision)
            .as_ref()
            .map_or(0, |caps| caps.max_vit_grid_tokens);
        merged.max_latent_feature_bytes = routed_caps(WorkVariant::EncodeLatent)
            .as_ref()
            .map_or(0, |caps| caps.max_latent_feature_bytes);
        merged.max_vision_feature_bytes = routed_caps(WorkVariant::EncodeVision)
            .as_ref()
            .map_or(0, |caps| caps.max_vision_feature_bytes);
        if let Some(materialize) = routed_caps(WorkVariant::Materialize) {
            merged.commit_marker_tokens = materialize.commit_marker_tokens;
            merged.gen_rope_advance = materialize.gen_rope_advance;
        }
        merged.encoder_cache_budget = [WorkVariant::EncodeLatent, WorkVariant::EncodeVision]
            .into_iter()
            .filter_map(|variant| routed_caps(variant).map(|caps| caps.encoder_cache_budget))
            .min()
            .unwrap_or(0);
        if let Some(decode) = routed_caps(WorkVariant::TokenDecode) {
            merged.adapter_mode = decode.adapter_mode;
        }
        Ok(merged)
    }

    fn pool_bit(index: usize) -> u64 {
        1u64 << index
    }

    fn cache_admissions(&mut self, admissions: &[Admission]) -> anyhow::Result<()> {
        for admission in admissions {
            if let Some(existing) = self.admissions.get(&admission.request_key) {
                anyhow::ensure!(
                    existing == admission,
                    "request {:?} was readmitted with a different descriptor before drop",
                    admission.request_key
                );
            } else {
                self.admissions
                    .insert(admission.request_key, admission.clone());
            }
        }
        Ok(())
    }

    fn admissions_for(
        &self,
        pool_index: usize,
        operations: &[Operation],
    ) -> anyhow::Result<Vec<Admission>> {
        let mut admissions = Vec::new();
        for operation in operations {
            if self
                .admitted_pools
                .contains(&(pool_index, operation.request_key))
            {
                continue;
            }
            let admission = self.admissions.get(&operation.request_key).ok_or_else(|| {
                anyhow::anyhow!(
                    "pool {pool_index} has no admission for request {:?}",
                    operation.request_key
                )
            })?;
            admissions.push(admission.clone());
        }
        Ok(admissions)
    }

    /// The pools a control command must reach. A commit/close/release rides with
    /// the operation that produced the version it names, so it targets the pool
    /// running this request's operation in this batch; a control for a request
    /// with no operation this batch is broadcast to every pool that has admitted
    /// the request.
    fn control_pools(&self, batch: &Batch, request_key: RequestKey) -> Vec<usize> {
        if let Some(pool_index) = batch
            .operations
            .iter()
            .find(|operation| operation.request_key == request_key)
            .and_then(|operation| self.routing.get(&operation.work.variant()).copied())
        {
            return vec![pool_index];
        }
        (0..self.pools.len())
            .filter(|pool_index| self.admitted_pools.contains(&(*pool_index, request_key)))
            .collect()
    }

    fn submit_partition(
        &mut self,
        pool_index: usize,
        step_id: u64,
        operations: Vec<Operation>,
        controls: Vec<Control>,
        input_products: Vec<ProductPayload>,
    ) -> anyhow::Result<bool> {
        if operations.is_empty() && controls.is_empty() {
            return Ok(false);
        }
        let admissions = self.admissions_for(pool_index, &operations)?;
        let request_keys = admissions
            .iter()
            .map(|admission| admission.request_key)
            .collect::<Vec<_>>();
        let partition = Batch::new(step_id, admissions, operations)
            .with_controls(controls)
            .with_input_products(input_products);
        partition.validate()?;
        self.pools[pool_index].exec.submit(partition)?;
        self.admitted_pools
            .extend(request_keys.into_iter().map(|key| (pool_index, key)));
        Ok(true)
    }

    fn pump(&mut self) -> anyhow::Result<()> {
        for pool_index in 0..self.pools.len() {
            loop {
                let report = self.pools[pool_index].exec.poll()?;
                let Some(report) = report else {
                    break;
                };
                self.route_result(pool_index, report)?;
            }
        }
        Ok(())
    }

    fn route_result(&mut self, pool_index: usize, report: CompletionReport) -> anyhow::Result<()> {
        let step_id = report.step_id;
        let pool_bit = Self::pool_bit(pool_index);
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
                slots.len() == report.completions.len(),
                "pool {pool_index} returned {} completions for step {step_id}, expected {}",
                report.completions.len(),
                slots.len()
            );
            for (slot, completion) in slots.into_iter().zip(report.completions) {
                anyhow::ensure!(
                    step.outputs[slot].is_none(),
                    "pool {pool_index} returned operation slot {slot} twice for step {step_id}"
                );
                let operation = &step.operations[slot];
                anyhow::ensure!(
                    completion.request_key == operation.request_key
                        && completion.op_id == operation.op_id,
                    "pool {pool_index} completion identity does not match operation slot {slot} for step {step_id}"
                );
                step.outputs[slot] = Some(completion);
            }
            step.expected_pools &= !pool_bit;
            step.products.extend(report.products);
            step.worker_exec_us = step
                .worker_exec_us
                .saturating_add(report.worker_exec_us.unwrap_or(0));
            // A step is registration-visible only when every contributing
            // partition reports its registration visible.
            step.registration_visible &= report.registration.visible;
            if step.forward_stats.is_none() {
                step.forward_stats = report.forward_stats;
            }
        }
        self.try_complete(step_id)
    }

    fn try_complete(&mut self, step_id: u64) -> anyhow::Result<()> {
        let complete = self
            .pending
            .get(&step_id)
            .is_some_and(|step| step.expected_pools == 0);
        if !complete {
            return Ok(());
        }
        let step = self
            .pending
            .remove(&step_id)
            .ok_or_else(|| anyhow::anyhow!("pending step {step_id} disappeared"))?;
        let completions = step
            .outputs
            .into_iter()
            .enumerate()
            .map(|(slot, output)| {
                output.ok_or_else(|| {
                    anyhow::anyhow!("stage merge missed operation slot {slot} for step {step_id}")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        self.ready.push_back(CompletionReport {
            step_id,
            completions,
            products: step.products,
            registration: RegistrationAck {
                visible: step.registration_visible,
            },
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
        self.admissions
            .retain(|request_key, _| request_key.session_id != session_id);
        self.admitted_pools
            .retain(|(_, request_key)| request_key.session_id != session_id);
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

    fn device_products_reachable(&self, producer: WorkVariant, consumer: WorkVariant) -> bool {
        self.routing.contains_key(&producer)
            && self.routing.get(&producer) == self.routing.get(&consumer)
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
            .collect::<Vec<Vec<Operation>>>();
        let mut partition_inputs = (0..self.pools.len())
            .map(|_| Vec::new())
            .collect::<Vec<Vec<ProductPayload>>>();
        let mut routes = Vec::with_capacity(batch.operations.len());
        let mut input_routes = HashMap::new();
        for operation in &batch.operations {
            let variant = operation.work.variant();
            let pool_index = self.routing.get(&variant).copied().ok_or_else(|| {
                anyhow::anyhow!(
                    "StageRouter has no pool for work variant {variant:?} in request {:?}",
                    operation.request_key
                )
            })?;
            // A generation flow routed to a pool other than the pool that runs
            // the request's token lineage consumes its conditioning across the
            // stage boundary; that dependency must be named by a declared input
            // product so the consuming pool can resolve it from the data plane.
            if variant == WorkVariant::GenFlow {
                let token_pool = self
                    .routing
                    .get(&WorkVariant::TokenDecode)
                    .or_else(|| self.routing.get(&WorkVariant::TokenExtend));
                if token_pool.is_some_and(|token_pool| *token_pool != pool_index) {
                    anyhow::ensure!(
                        !operation.inputs.is_empty(),
                        "cross-pool generation flow for request {:?} names no input product for its conditioning",
                        operation.request_key
                    );
                }
            }
            for input in &operation.inputs {
                input_routes.insert(input.clone(), pool_index);
            }
            routes.push(pool_index);
            partitions[pool_index].push(operation.clone());
        }
        for payload in &batch.input_products {
            let pool_index = input_routes.get(&payload.product).copied().ok_or_else(|| {
                anyhow::anyhow!(
                    "StageRouter cannot route undeclared input product {:?}",
                    payload.product
                )
            })?;
            partition_inputs[pool_index].push(payload.clone());
        }

        let mut pool_controls = (0..self.pools.len())
            .map(|_| Vec::new())
            .collect::<Vec<Vec<Control>>>();
        for control in &batch.controls {
            let request_key = control.request_key();
            let targets = self.control_pools(&batch, request_key);
            anyhow::ensure!(
                !targets.is_empty(),
                "stage router received a control for request {request_key:?} that is not admitted to any pool"
            );
            for pool_index in targets {
                pool_controls[pool_index].push(control.clone());
            }
        }

        let mut expected_pools = 0;
        for (pool_index, ((operations, controls), input_products)) in partitions
            .into_iter()
            .zip(pool_controls)
            .zip(partition_inputs)
            .enumerate()
        {
            if self.submit_partition(
                pool_index,
                batch.step_id,
                operations,
                controls,
                input_products,
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
                products: Vec::new(),
                registration_visible: true,
                worker_exec_us: 0,
                forward_stats: None,
            },
        );
        Ok(())
    }

    fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
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

    fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use uniserve_core::{RequestId, SamplingParams};
    use uniserve_worker_wire::{
        Bounds, DType, DimBound, Domain, KvAllocation, OpId, PointRange, ProductKind, ProductRef,
        RouteId, ShapeBound, StorageClass, TokenMode, UndAdmission, VersionRef, Work,
        encode_token_product_bytes,
    };

    use super::*;

    struct RecordingExecutor {
        caps: EngineCaps,
        submissions: Arc<Mutex<Vec<Batch>>>,
    }

    impl Executor for RecordingExecutor {
        fn caps(&self) -> EngineCaps {
            self.caps.clone()
        }

        fn pipeline_depth(&self) -> usize {
            1
        }

        fn in_flight(&self) -> usize {
            0
        }

        fn submit(&mut self, batch: Batch) -> anyhow::Result<()> {
            batch.validate()?;
            self.submissions.lock().unwrap().push(batch);
            Ok(())
        }

        fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
            Ok(None)
        }

        fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
            anyhow::bail!("recording executor has no completion source")
        }

        fn control(&mut self, _operation: ControlOp) -> anyhow::Result<u64> {
            Ok(1)
        }

        fn control_wait(
            &mut self,
            _operation: ControlOp,
            _targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            Ok(Vec::new())
        }
    }

    fn caps(work: WorkVariant) -> EngineCaps {
        let mut caps = EngineCaps {
            supported_work: vec![work],
            max_vit_grid_tokens: 64,
            max_vision_feature_bytes: 1 << 20,
            ..EngineCaps::default()
        };
        caps.route_capability_digest = caps.compute_route_capability_digest();
        caps
    }

    fn request_key(session_id: u64) -> RequestKey {
        RequestKey::new(1, RequestId(session_id), 1)
    }

    fn admission(request_key: RequestKey) -> Admission {
        Admission::new(
            request_key,
            Some(UndAdmission {
                sampling: SamplingParams::default(),
                negative_token_ids: Vec::new(),
                kv: KvAllocation::default(),
            }),
            None,
            None,
        )
        .unwrap()
    }

    fn host_input(
        request_key: RequestKey,
        op_id: OpId,
        generation: u32,
        kind: ProductKind,
        dtype: DType,
        elements: u32,
    ) -> ProductRef {
        ProductRef {
            request_key,
            producer_op_id: op_id,
            output_index: 0,
            generation,
            kind,
            storage_class: StorageClass::HostStaging,
            dtype,
            shape_bound: ShapeBound {
                dims: vec![DimBound::Static(elements)],
            },
            point_range: PointRange::default(),
        }
    }

    #[test]
    fn host_inputs_follow_their_consuming_operations_to_each_stage() {
        let encoder_submissions = Arc::new(Mutex::new(Vec::new()));
        let prefill_submissions = Arc::new(Mutex::new(Vec::new()));
        let encoder = RecordingExecutor {
            caps: caps(WorkVariant::EncodeVision),
            submissions: Arc::clone(&encoder_submissions),
        };
        let prefill = RecordingExecutor {
            caps: caps(WorkVariant::TokenExtend),
            submissions: Arc::clone(&prefill_submissions),
        };
        let mut router = StageRouter::try_new(vec![
            (WorkerKind::Encoder, Box::new(encoder)),
            (WorkerKind::Prefill, Box::new(prefill)),
        ])
        .unwrap();

        let encode_key = request_key(11);
        let encode_op_id = OpId(21);
        let image = host_input(
            encode_key,
            encode_op_id,
            31,
            ProductKind::Artifact,
            DType::U8,
            4,
        );
        let encode = Operation::registered(
            encode_key,
            encode_op_id,
            VersionRef::admission_root(encode_key, OpId(1), "0".repeat(64)),
            Work::Encode(uniserve_worker_wire::EncodeMode::Vision),
            RouteId(0),
            Domain::Und,
            Bounds::default(),
            vec![image.clone()],
            Vec::new(),
            Vec::new(),
            None,
            None,
            0,
        );

        let prefill_key = request_key(12);
        let prefill_op_id = OpId(22);
        let tokens = host_input(
            prefill_key,
            prefill_op_id,
            32,
            ProductKind::Token,
            DType::U32,
            3,
        );
        let prefill = Operation::registered(
            prefill_key,
            prefill_op_id,
            VersionRef::admission_root(prefill_key, OpId(1), "1".repeat(64)),
            Work::Token(TokenMode::Extend),
            RouteId(0),
            Domain::Und,
            Bounds {
                max_points: 1,
                max_tokens: 3,
                ..Bounds::default()
            },
            vec![tokens.clone()],
            Vec::new(),
            Vec::new(),
            None,
            None,
            0,
        );

        router
            .submit(
                Batch::new(
                    7,
                    vec![admission(encode_key), admission(prefill_key)],
                    vec![encode, prefill],
                )
                .with_input_products(vec![
                    ProductPayload {
                        product: image.clone(),
                        bytes: vec![1, 2, 3, 4],
                    },
                    ProductPayload {
                        product: tokens.clone(),
                        bytes: encode_token_product_bytes(&[7, 8, 9]),
                    },
                ]),
            )
            .unwrap();

        let encoder_batches = encoder_submissions.lock().unwrap();
        assert_eq!(encoder_batches.len(), 1);
        assert_eq!(
            encoder_batches[0].operations[0].work.variant(),
            WorkVariant::EncodeVision
        );
        assert_eq!(
            encoder_batches[0].input_products,
            vec![ProductPayload {
                product: image,
                bytes: vec![1, 2, 3, 4],
            }]
        );

        let prefill_batches = prefill_submissions.lock().unwrap();
        assert_eq!(prefill_batches.len(), 1);
        assert_eq!(
            prefill_batches[0].operations[0].work.variant(),
            WorkVariant::TokenExtend
        );
        assert_eq!(
            prefill_batches[0].input_products,
            vec![ProductPayload {
                product: tokens,
                bytes: encode_token_product_bytes(&[7, 8, 9]),
            }]
        );
        assert!(
            router.device_products_reachable(WorkVariant::EncodeVision, WorkVariant::EncodeVision)
        );
    }
}
