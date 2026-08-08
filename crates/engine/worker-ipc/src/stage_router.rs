//! Typed control-plane routing across heterogeneous worker pools.
//!
//! A staged topology partitions each execution batch by exact [`WorkVariant`],
//! sends every session admission to a pool before that pool's first operation
//! for the session, and publishes each independently ready partition by identity.
//! A worker-local device product is directly reachable
//! only within its producing pool. Cross-pool movement requires a declared
//! `Transfer(Product)` whose destination reference belongs to the consumer.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::time::Duration;

use anyhow::Context;
use uniserve_core::{CommandWaker, RequestId};
use uniserve_executor::{ControlAck, ControlOp, Executor, WorkerKind};
use uniserve_worker_wire::{
    Admission, Batch, BatchPartition, CompletionReport, Control, EngineCaps, Operation,
    ProductPayload, ProductRef, RequestKey, RouteExecutionCapability, RouteId,
    TRANSFER_DESCRIPTOR_PREFIX, WorkVariant, is_transfer_descriptor,
};

struct PoolEntry {
    exec: Box<dyn Executor>,
}

struct PendingStep {
    expected_pools: u64,
    partitions: Vec<BatchPartition>,
    partition_routes: HashMap<u32, usize>,
    returned_partitions: HashSet<u32>,
}

#[derive(Clone)]
struct ProductRoute {
    pool_index: usize,
    producer_plan_digest: String,
}

fn transfer_identity(bytes: &[u8]) -> anyhow::Result<(String, String)> {
    anyhow::ensure!(
        is_transfer_descriptor(bytes),
        "cross-stage product has no transfer descriptor frame"
    );
    let encoded = &bytes[TRANSFER_DESCRIPTOR_PREFIX.len()..];
    let value: serde_json::Value = serde_json::from_slice(encoded)?;
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("cross-stage transfer descriptor is not an object"))?;
    anyhow::ensure!(
        object.len() == 3
            && object.contains_key("kind")
            && object.contains_key("producer_plan_digest")
            && object.contains_key("value"),
        "cross-stage transfer descriptor has an invalid shape"
    );
    let kind = object
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("cross-stage transfer descriptor has no kind"))?;
    anyhow::ensure!(
        matches!(kind, "tensor" | "kv")
            && object
                .get("value")
                .is_some_and(serde_json::Value::is_object),
        "cross-stage transfer descriptor has an invalid kind or value"
    );
    let digest = object
        .get("producer_plan_digest")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("cross-stage transfer descriptor has no plan digest"))?;
    anyhow::ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "cross-stage transfer descriptor plan digest is invalid"
    );
    anyhow::ensure!(
        serde_json::to_vec(&value)? == encoded,
        "cross-stage transfer descriptor is not canonical JSON"
    );
    Ok((kind.to_string(), digest.to_string()))
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
    operation_routes: HashMap<(RequestKey, uniserve_worker_wire::OpId), usize>,
    product_routes: HashMap<ProductRef, ProductRoute>,
    transfer_products: HashMap<ProductRef, ProductPayload>,
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
            caps.validate()
                .with_context(|| format!("staged pool {index} reported invalid capabilities"))?;
            anyhow::ensure!(
                executor.pipeline_depth() == caps.pipeline_depth.max(1) as usize,
                "staged pool {index} executor depth {} disagrees with capability depth {}",
                executor.pipeline_depth(),
                caps.pipeline_depth.max(1),
            );
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
            operation_routes: HashMap::new(),
            product_routes: HashMap::new(),
            transfer_products: HashMap::new(),
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
        merged.execution_constraints.max_speculative_points = pools
            .iter()
            .map(|pool| {
                pool.exec
                    .caps()
                    .execution_constraints
                    .max_speculative_points
            })
            .min()
            .unwrap_or(1);
        merged.execution_constraints.device_sequence_lengths = pools.iter().all(|pool| {
            pool.exec
                .caps()
                .execution_constraints
                .device_sequence_lengths
        });
        merged.execution_constraints.device_append_offsets = pools
            .iter()
            .all(|pool| pool.exec.caps().execution_constraints.device_append_offsets);
        merged.execution_constraints.incremental_kv_publication = pools.iter().all(|pool| {
            pool.exec
                .caps()
                .execution_constraints
                .incremental_kv_publication
        });
        let mut route_capabilities: HashMap<RouteId, RouteExecutionCapability> = HashMap::new();
        for pool in pools {
            for capability in pool.exec.caps().execution_constraints.route_capabilities {
                match route_capabilities.get_mut(&capability.route) {
                    Some(existing) => {
                        anyhow::ensure!(
                            existing.sampling_ownership == capability.sampling_ownership,
                            "staged pools disagree on sampling ownership for route {}",
                            capability.route.0
                        );
                        existing.tensorized_mixed &= capability.tensorized_mixed;
                        existing.preemptible &= capability.preemptible;
                        extend_unique(&mut existing.supported_work, capability.supported_work);
                        existing.credits.per_request = existing
                            .credits
                            .per_request
                            .checked_add(capability.credits.per_request)
                            .context("staged route per-request credit sum overflowed")?;
                    }
                    None => {
                        route_capabilities.insert(capability.route, capability);
                    }
                }
            }
        }
        let aggregate_worker_credits =
            pools
                .iter()
                .try_fold(uniserve_worker_wire::CreditVector::ZERO, |total, pool| {
                    let capacity = pool
                        .exec
                        .caps()
                        .execution_constraints
                        .route_capabilities
                        .first()
                        .map(|capability| capability.credits.worker)
                        .context("staged pool has no route credit capacity")?;
                    total
                        .checked_add(capacity)
                        .context("staged worker credit sum overflowed")
                })?;
        for capability in route_capabilities.values_mut() {
            let owning_pools = capability
                .supported_work
                .iter()
                .filter_map(|variant| routing.get(variant).copied())
                .collect::<HashSet<_>>();
            capability.tensorized_mixed &= owning_pools.len() == 1;
            capability.preemptible &= owning_pools.len() == 1;
            capability.credits.worker = aggregate_worker_credits;
        }
        let mut route_capabilities = route_capabilities.into_values().collect::<Vec<_>>();
        route_capabilities.sort_unstable_by_key(|capability| capability.route.0);
        merged.execution_constraints.route_capabilities = route_capabilities;

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
        merged.restored_snapshots.clear();
        merged.validate()?;
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
    fn control_pools(&self, control: &Control) -> Vec<usize> {
        let (request_key, producer_op_id) = match control {
            Control::Commit {
                request_key,
                selected,
                ..
            } => (*request_key, selected.producer_op_id),
            Control::Close {
                request_key,
                cutoff,
                ..
            } => (*request_key, cutoff.producer_op_id),
            Control::Release { request_key, op_id } => (*request_key, *op_id),
        };
        if let Some(pool_index) = self
            .operation_routes
            .get(&(request_key, producer_op_id))
            .copied()
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
        partitions: Vec<BatchPartition>,
        controls: Vec<Control>,
        input_products: Vec<ProductPayload>,
    ) -> anyhow::Result<bool> {
        if partitions.is_empty() && controls.is_empty() {
            return Ok(false);
        }
        let operations = partitions
            .iter()
            .flat_map(|partition| partition.operations.iter())
            .cloned()
            .collect::<Vec<_>>();
        let admissions = self.admissions_for(pool_index, &operations)?;
        let request_keys = admissions
            .iter()
            .map(|admission| admission.request_key)
            .collect::<Vec<_>>();
        let partition = Batch::new(step_id, admissions, partitions)
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
        report.validate()?;
        let mut report = report;
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
            let expected_partition_ids = step
                .partition_routes
                .iter()
                .filter_map(|(partition_id, route)| (*route == pool_index).then_some(*partition_id))
                .collect::<HashSet<_>>();
            anyhow::ensure!(
                !report.partitions.is_empty() || expected_partition_ids.is_empty(),
                "pool {pool_index} returned an empty partial report for step {step_id}"
            );
            for partition in &report.partitions {
                anyhow::ensure!(
                    expected_partition_ids.contains(&partition.partition_id),
                    "pool {pool_index} returned unexpected partition {} for step {step_id}",
                    partition.partition_id
                );
                let planned = step
                    .partitions
                    .iter()
                    .find(|planned| planned.partition_id == partition.partition_id)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "partition {} disappeared from pending step {step_id}",
                            partition.partition_id
                        )
                    })?;
                anyhow::ensure!(
                    planned.operations.len() == partition.completions.len()
                        && planned.operations.iter().zip(&partition.completions).all(
                            |(operation, completion)| {
                                operation.request_key == completion.request_key
                                    && operation.op_id == completion.op_id
                            }
                        ),
                    "pool {pool_index} completion identities do not match partition {} for step {step_id}",
                    partition.partition_id
                );
                anyhow::ensure!(
                    step.returned_partitions.insert(partition.partition_id),
                    "pool {pool_index} returned a partition twice for step {step_id}"
                );
            }
            if expected_partition_ids
                .iter()
                .all(|partition_id| step.returned_partitions.contains(partition_id))
            {
                step.expected_pools &= !pool_bit;
            }
        }
        for partition in &mut report.partitions {
            let mut visible_products = Vec::with_capacity(partition.products.len());
            for product in partition.products.drain(..) {
                if !is_transfer_descriptor(&product.bytes) {
                    visible_products.push(product);
                    continue;
                }
                let route = self.product_routes.get(&product.product).ok_or_else(|| {
                    anyhow::anyhow!(
                        "pool {pool_index} returned an unplanned cross-stage product {:?}",
                        product.product
                    )
                })?;
                anyhow::ensure!(
                    route.pool_index == pool_index,
                    "pool {pool_index} returned a cross-stage product owned by pool {}",
                    route.pool_index
                );
                let (kind, producer_plan_digest) = transfer_identity(&product.bytes)?;
                anyhow::ensure!(
                    producer_plan_digest == route.producer_plan_digest,
                    "cross-stage product plan digest conflicts with its producer"
                );
                if let Some(existing) = self.transfer_products.get(&product.product) {
                    anyhow::ensure!(
                        existing == &product,
                        "cross-stage product identity was reused with conflicting transport bytes"
                    );
                } else {
                    self.transfer_products
                        .insert(product.product.clone(), product.clone());
                }
                if kind == "kv" {
                    visible_products.push(ProductPayload {
                        product: product.product,
                        bytes: Vec::new(),
                    });
                }
            }
            partition.products = visible_products;
        }
        if !report.partitions.is_empty() {
            self.ready.push_back(report);
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
        anyhow::ensure!(
            step.returned_partitions.len() == step.partitions.len(),
            "stage execution finished without every planned partition"
        );
        if step.partitions.is_empty() {
            self.ready.push_back(CompletionReport {
                step_id,
                partitions: Vec::new(),
            });
        }
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
        self.operation_routes
            .retain(|(request_key, _), _| request_key.session_id != session_id);
        self.product_routes
            .retain(|product, _| product.request_key.session_id != session_id);
        self.transfer_products
            .retain(|product, _| product.request_key.session_id != session_id);
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

        let mut pool_partitions = (0..self.pools.len())
            .map(|_| Vec::new())
            .collect::<Vec<Vec<BatchPartition>>>();
        let mut partition_inputs = (0..self.pools.len())
            .map(|_| Vec::new())
            .collect::<Vec<Vec<ProductPayload>>>();
        let mut partition_routes = HashMap::with_capacity(batch.partitions.len());
        let mut input_routes = HashMap::new();
        let mut group_pools = HashMap::new();
        for partition in &batch.partitions {
            let mut pool_index = None;
            for operation in &partition.operations {
                let variant = operation.work.variant();
                let operation_pool = self.routing.get(&variant).copied().ok_or_else(|| {
                    anyhow::anyhow!(
                        "StageRouter has no pool for work variant {variant:?} in request {:?}",
                        operation.request_key
                    )
                })?;
                if let Some(existing) = self
                    .operation_routes
                    .insert((operation.request_key, operation.op_id), operation_pool)
                {
                    anyhow::ensure!(
                        existing == operation_pool,
                        "operation identity was routed to conflicting staged pools"
                    );
                }
                for output in &operation.outputs {
                    let route = ProductRoute {
                        pool_index: operation_pool,
                        producer_plan_digest: operation.plan_digest.clone(),
                    };
                    if let Some(existing) = self.product_routes.get(output) {
                        anyhow::ensure!(
                            existing.pool_index == route.pool_index
                                && existing.producer_plan_digest == route.producer_plan_digest,
                            "product identity was routed with conflicting producer provenance"
                        );
                    } else {
                        self.product_routes.insert(output.clone(), route);
                    }
                }
                anyhow::ensure!(
                    pool_index.is_none_or(|index| index == operation_pool),
                    "partition {} spans staged pools; scheduler partition ownership is not executable",
                    partition.partition_id
                );
                pool_index = Some(operation_pool);
                if variant == WorkVariant::GenFlow {
                    let token_pool = self
                        .routing
                        .get(&WorkVariant::TokenDecode)
                        .or_else(|| self.routing.get(&WorkVariant::TokenExtend));
                    if token_pool.is_some_and(|token_pool| *token_pool != operation_pool) {
                        anyhow::ensure!(
                            !operation.inputs.is_empty(),
                            "cross-pool generation flow for request {:?} names no input product for its conditioning",
                            operation.request_key
                        );
                    }
                }
                for input in &operation.inputs {
                    anyhow::ensure!(
                        input_routes
                            .insert(input.clone(), operation_pool)
                            .is_none_or(|existing| existing == operation_pool),
                        "one input product is consumed across staged pools"
                    );
                }
            }
            let Some(pool_index) = pool_index else {
                anyhow::bail!(
                    "validated partition {} has no routed operation",
                    partition.partition_id
                );
            };
            anyhow::ensure!(
                group_pools
                    .insert(partition.submission_group, pool_index)
                    .is_none_or(|existing| existing == pool_index),
                "submission group {} spans staged pools and cannot share one physical runner call",
                partition.submission_group
            );
            partition_routes.insert(partition.partition_id, pool_index);
            pool_partitions[pool_index].push(partition.clone());
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
        let mut cross_pool_consumers = HashSet::new();
        for partition in &batch.partitions {
            let consumer_pool = partition_routes[&partition.partition_id];
            for operation in &partition.operations {
                for input in &operation.inputs {
                    if input.storage_class == uniserve_worker_wire::StorageClass::HostStaging {
                        continue;
                    }
                    let producer = self.product_routes.get(input).ok_or_else(|| {
                        anyhow::anyhow!(
                            "cross-stage input {:?} has no exact producer provenance",
                            input
                        )
                    })?;
                    if producer.pool_index == consumer_pool {
                        continue;
                    }
                    let payload = self.transfer_products.get(input).ok_or_else(|| {
                        anyhow::anyhow!("cross-stage input {:?} is not producer-ready", input)
                    })?;
                    anyhow::ensure!(
                        !partition_inputs[consumer_pool]
                            .iter()
                            .any(|existing| existing.product == *input),
                        "cross-stage input payload is supplied more than once"
                    );
                    partition_inputs[consumer_pool].push(payload.clone());
                    cross_pool_consumers.insert((input.request_key, input.producer_op_id));
                }
            }
        }

        let mut pool_controls = (0..self.pools.len())
            .map(|_| Vec::new())
            .collect::<Vec<Vec<Control>>>();
        for control in &batch.controls {
            let request_key = control.request_key();
            if let Control::Release { request_key, op_id } = control {
                anyhow::ensure!(
                    !cross_pool_consumers.contains(&(*request_key, *op_id)),
                    "a cross-stage producer cannot be released in its consumer submission"
                );
            }
            let targets = self.control_pools(control);
            anyhow::ensure!(
                !targets.is_empty(),
                "stage router received a control for request {request_key:?} that is not admitted to any pool"
            );
            for pool_index in targets {
                pool_controls[pool_index].push(control.clone());
            }
        }

        let mut expected_pools = 0;
        for (pool_index, ((partitions, controls), input_products)) in pool_partitions
            .into_iter()
            .zip(pool_controls)
            .zip(partition_inputs)
            .enumerate()
        {
            if self.submit_partition(
                pool_index,
                batch.step_id,
                partitions,
                controls,
                input_products,
            )? {
                expected_pools |= Self::pool_bit(pool_index);
            }
        }
        self.pending.insert(
            batch.step_id,
            PendingStep {
                expected_pools,
                partitions: batch.partitions,
                partition_routes,
                returned_partitions: HashSet::new(),
            },
        );
        for control in &batch.controls {
            if let Control::Release { request_key, op_id } = control {
                self.transfer_products.retain(|product, _| {
                    product.request_key != *request_key || product.producer_op_id != *op_id
                });
                self.product_routes.retain(|product, _| {
                    product.request_key != *request_key || product.producer_op_id != *op_id
                });
            }
        }
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
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use uniserve_core::{RequestId, SamplingParams};
    use uniserve_worker_wire::{
        AttentionRegime, Bounds, DType, DimBound, Domain, ExecutionCapability, KvAdmission, OpId,
        OpStatus, PointRange, ProductKind, ProductRef, RegistrationAck, RouteId, ShapeBound,
        StorageClass, TokenMode, UndAdmission, VersionRef, Work, encode_token_product_bytes,
    };

    use super::*;

    struct RecordingExecutor {
        caps: EngineCaps,
        submissions: Arc<Mutex<Vec<Batch>>>,
    }

    struct ReportingExecutor {
        caps: EngineCaps,
        submissions: Arc<Mutex<Vec<Batch>>>,
        reports: Arc<Mutex<VecDeque<CompletionReport>>>,
    }

    impl Executor for ReportingExecutor {
        fn caps(&self) -> EngineCaps {
            self.caps.clone()
        }

        fn pipeline_depth(&self) -> usize {
            self.caps.pipeline_depth.max(1) as usize
        }

        fn in_flight(&self) -> usize {
            self.reports.lock().unwrap().len()
        }

        fn submit(&mut self, batch: Batch) -> anyhow::Result<()> {
            batch.validate()?;
            self.submissions.lock().unwrap().push(batch);
            Ok(())
        }

        fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
            Ok(self.reports.lock().unwrap().pop_front())
        }

        fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
            self.poll()?
                .ok_or_else(|| anyhow::anyhow!("reporting executor has no completion"))
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

    impl Executor for RecordingExecutor {
        fn caps(&self) -> EngineCaps {
            self.caps.clone()
        }

        fn pipeline_depth(&self) -> usize {
            self.caps.pipeline_depth.max(1) as usize
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
        EngineCaps {
            supported_work: vec![work],
            execution_constraints: uniserve_worker_wire::ExecutionConstraints {
                route_capabilities: vec![RouteExecutionCapability {
                    route: RouteId(0),
                    supported_work: vec![work],
                    tensorized_mixed: false,
                    sampling_ownership: uniserve_worker_wire::SamplingOwnership::DesignatedRank,
                    preemptible: false,
                    credits: uniserve_worker_wire::ExecutionConstraints::default()
                        .route_capabilities[0]
                        .credits,
                    max_unresolved_window: 1,
                    legal_feature_bitset: 0,
                    sampler_processors: 0,
                    processor_order_revision: 1,
                    rng_layouts: 1,
                    graph_eligible: false,
                    gen_conditioning: 0,
                    max_points_per_operation: 1,
                    mixed_row_combinations: Vec::new(),
                }],
                ..Default::default()
            },
            max_vit_grid_tokens: 64,
            max_vision_feature_bytes: 1 << 20,
            ..EngineCaps::default()
        }
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
                finish_token_ids: Vec::new(),
                kv: KvAdmission::default(),
            }),
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

    fn feature_product(request_key: RequestKey, op_id: OpId) -> ProductRef {
        ProductRef {
            request_key,
            producer_op_id: op_id,
            output_index: 0,
            generation: 41,
            kind: ProductKind::VisionFeature,
            storage_class: StorageClass::LatentArena,
            dtype: DType::F32,
            shape_bound: ShapeBound {
                dims: vec![DimBound::Static(4)],
            },
            point_range: PointRange::default(),
        }
    }

    fn transfer_descriptor(plan_digest: &str) -> Vec<u8> {
        let value = serde_json::json!({
            "kind": "tensor",
            "producer_plan_digest": plan_digest,
            "value": {
                "height": 16,
                "locator": {
                    "device": "cuda:0",
                    "dtype": "float32",
                    "handle_b64": "",
                    "meta": {},
                    "nbytes": 16,
                    "session": "producer",
                    "shape": [4],
                    "transport": "cuda_ipc",
                    "version": 1
                },
                "payload_kind": "vision_feature",
                "width": 16
            }
        });
        let mut bytes = TRANSFER_DESCRIPTOR_PREFIX.to_vec();
        bytes.extend(serde_json::to_vec(&value).unwrap());
        bytes
    }

    fn completion_report(
        step_id: u64,
        partition_id: u32,
        operation: &Operation,
        product: ProductPayload,
    ) -> CompletionReport {
        CompletionReport {
            step_id,
            partitions: vec![uniserve_worker_wire::PartitionCompletion {
                partition_id,
                completions: vec![uniserve_worker_wire::CompletionRecord {
                    request_key: operation.request_key,
                    op_id: operation.op_id,
                    completion_slot_generation: 1,
                    status: OpStatus::Ok,
                    selected_point: 0,
                    logical_lengths: Default::default(),
                    token_span: Default::default(),
                    committed_tokens: Vec::new(),
                    finish_flags: Default::default(),
                    product_generations: vec![product.product.generation],
                    semantic_digest: "2".repeat(64),
                    error_code: None,
                    timing_counters: Default::default(),
                }],
                products: vec![product],
                registration: RegistrationAck { visible: true },
                worker_exec_us: Some(1),
                forward_stats: Default::default(),
            }],
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
            0,
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
            0,
            None,
            None,
            0,
        );

        router
            .submit(
                Batch::new(
                    7,
                    vec![admission(encode_key), admission(prefill_key)],
                    vec![
                        BatchPartition {
                            partition_id: 1,
                            submission_group: 1,
                            collective_seq: 1,
                            domain: Domain::Und,
                            route: RouteId(0),
                            execution: ExecutionCapability::DomainHomogeneous,
                            attention: AttentionRegime::None,
                            shape_class: 0,
                            operations: vec![encode],
                            kv_reservations: Vec::new(),
                        },
                        BatchPartition {
                            partition_id: 2,
                            submission_group: 2,
                            collective_seq: 2,
                            domain: Domain::Und,
                            route: RouteId(0),
                            execution: ExecutionCapability::DomainHomogeneous,
                            attention: AttentionRegime::Causal,
                            shape_class: 0,
                            operations: vec![prefill],
                            kv_reservations: Vec::new(),
                        },
                    ],
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
            encoder_batches[0]
                .operations()
                .next()
                .unwrap()
                .work
                .variant(),
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
            prefill_batches[0]
                .operations()
                .next()
                .unwrap()
                .work
                .variant(),
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

    #[test]
    fn exact_cross_stage_product_is_gated_stored_and_injected() {
        let encoder_submissions = Arc::new(Mutex::new(Vec::new()));
        let consumer_submissions = Arc::new(Mutex::new(Vec::new()));
        let encoder_reports = Arc::new(Mutex::new(VecDeque::new()));
        let consumer_reports = Arc::new(Mutex::new(VecDeque::new()));
        let encoder = ReportingExecutor {
            caps: caps(WorkVariant::EncodeVision),
            submissions: Arc::clone(&encoder_submissions),
            reports: Arc::clone(&encoder_reports),
        };
        let consumer = ReportingExecutor {
            caps: caps(WorkVariant::TokenExtend),
            submissions: Arc::clone(&consumer_submissions),
            reports: Arc::clone(&consumer_reports),
        };
        let mut router = StageRouter::try_new(vec![
            (WorkerKind::Encoder, Box::new(encoder)),
            (WorkerKind::Prefill, Box::new(consumer)),
        ])
        .unwrap();
        let key = request_key(91);
        let encode_op_id = OpId(11);
        let image = host_input(key, encode_op_id, 40, ProductKind::Artifact, DType::U8, 4);
        let feature = feature_product(key, encode_op_id);
        let encode = Operation::registered(
            key,
            encode_op_id,
            VersionRef::admission_root(key, OpId(1), "0".repeat(64)),
            Work::Encode(uniserve_worker_wire::EncodeMode::Vision),
            RouteId(0),
            Domain::Und,
            Bounds {
                max_points: 1,
                max_latent_bytes: 16,
                ..Bounds::default()
            },
            vec![image.clone()],
            vec![feature.clone()],
            0,
            None,
            None,
            0,
        );
        let payload = ProductPayload {
            product: feature.clone(),
            bytes: transfer_descriptor(&encode.plan_digest),
        };
        let encode_report = completion_report(1, 1, &encode, payload.clone());
        router
            .submit(
                Batch::new(
                    1,
                    vec![admission(key)],
                    vec![BatchPartition {
                        partition_id: 1,
                        submission_group: 1,
                        collective_seq: 1,
                        domain: Domain::Und,
                        route: RouteId(0),
                        execution: ExecutionCapability::DomainHomogeneous,
                        attention: AttentionRegime::None,
                        shape_class: 0,
                        operations: vec![encode],
                        kv_reservations: Vec::new(),
                    }],
                )
                .with_input_products(vec![ProductPayload {
                    product: image,
                    bytes: vec![1, 2, 3, 4],
                }]),
            )
            .unwrap();
        let consume = Operation::registered(
            key,
            OpId(12),
            VersionRef::admission_root(key, OpId(1), "0".repeat(64)),
            Work::Token(TokenMode::Extend),
            RouteId(0),
            Domain::Und,
            Bounds {
                max_points: 1,
                max_tokens: 1,
                ..Bounds::default()
            },
            vec![feature.clone()],
            Vec::new(),
            0,
            None,
            None,
            0,
        );
        let consumer_batch = Batch::new(
            2,
            Vec::new(),
            vec![BatchPartition {
                partition_id: 2,
                submission_group: 2,
                collective_seq: 2,
                domain: Domain::Und,
                route: RouteId(0),
                execution: ExecutionCapability::DomainHomogeneous,
                attention: AttentionRegime::Causal,
                shape_class: 0,
                operations: vec![consume],
                kv_reservations: Vec::new(),
            }],
        );
        let pending_error = router
            .submit(consumer_batch.clone())
            .unwrap_err()
            .to_string();
        assert!(
            pending_error.contains("not producer-ready"),
            "got: {pending_error}"
        );

        encoder_reports.lock().unwrap().push_back(encode_report);
        let producer_report = router.poll().unwrap().unwrap();
        assert!(producer_report.products().next().is_none());

        router.submit(consumer_batch).unwrap();
        let batches = consumer_submissions.lock().unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].input_products, vec![payload]);
    }

    #[test]
    fn cross_stage_product_rejects_conflicting_producer_digest() {
        let submissions = Arc::new(Mutex::new(Vec::new()));
        let reports = Arc::new(Mutex::new(VecDeque::new()));
        let encoder = ReportingExecutor {
            caps: caps(WorkVariant::EncodeVision),
            submissions,
            reports: Arc::clone(&reports),
        };
        let mut router =
            StageRouter::try_new(vec![(WorkerKind::Encoder, Box::new(encoder))]).unwrap();
        let key = request_key(92);
        let op_id = OpId(13);
        let image = host_input(key, op_id, 43, ProductKind::Artifact, DType::U8, 4);
        let feature = feature_product(key, op_id);
        let encode = Operation::registered(
            key,
            op_id,
            VersionRef::admission_root(key, OpId(1), "0".repeat(64)),
            Work::Encode(uniserve_worker_wire::EncodeMode::Vision),
            RouteId(0),
            Domain::Und,
            Bounds {
                max_points: 1,
                max_latent_bytes: 16,
                ..Bounds::default()
            },
            vec![image.clone()],
            vec![feature.clone()],
            0,
            None,
            None,
            0,
        );
        let report = completion_report(
            3,
            3,
            &encode,
            ProductPayload {
                product: feature,
                bytes: transfer_descriptor(&"f".repeat(64)),
            },
        );
        router
            .submit(
                Batch::new(
                    3,
                    vec![admission(key)],
                    vec![BatchPartition {
                        partition_id: 3,
                        submission_group: 3,
                        collective_seq: 3,
                        domain: Domain::Und,
                        route: RouteId(0),
                        execution: ExecutionCapability::DomainHomogeneous,
                        attention: AttentionRegime::None,
                        shape_class: 0,
                        operations: vec![encode],
                        kv_reservations: Vec::new(),
                    }],
                )
                .with_input_products(vec![ProductPayload {
                    product: image,
                    bytes: vec![1, 2, 3, 4],
                }]),
            )
            .unwrap();
        reports.lock().unwrap().push_back(report);

        let error = router.poll().unwrap_err().to_string();
        assert!(error.contains("plan digest conflicts with its producer"));
    }
}
