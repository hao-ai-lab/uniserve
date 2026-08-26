//! Typed control-plane routing across heterogeneous worker pools.
//!
//! A staged topology partitions each execution batch by exact [`ForwardMode`],
//! sends every session admission to a pool before that pool's first operation
//! for the session, and publishes each independently ready partition by identity.
//! A worker-local device product remains resident within its producing pool.
//! The executor retains a cross-pool consumer by exact product identity until the
//! producer publishes the bounded transfer descriptor, then submits the
//! consumer before exposing the producer completion to the scheduler.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::time::Duration;

use crate::executor::{ControlAck, ControlOp, Executor, WorkerKind};
use anyhow::Context;
use uniserve_core::{CommandWaker, RequestId};
use uniserve_worker_ipc::{
    Admission, Batch, BatchPartition, CompletionReport, Control, ForwardMode, Operation,
    ProductPayload, ProductRef, RequestKey, TRANSFER_DESCRIPTOR_PREFIX, WorkerCapabilities,
    is_transfer_descriptor,
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

struct PoolSubmission {
    step_id: u64,
    partitions: Vec<BatchPartition>,
    controls: Vec<Control>,
    input_products: Vec<ProductPayload>,
    dependencies: Vec<ProductRef>,
    collective_seqs: Vec<u64>,
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
        matches!(kind, "encoder" | "device_product" | "latent" | "kv")
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
pub struct StagedExecutor {
    routing: HashMap<ForwardMode, usize>,
    pools: Vec<PoolEntry>,
    caps: WorkerCapabilities,
    depth: usize,
    pending: BTreeMap<u64, PendingStep>,
    ready: VecDeque<CompletionReport>,
    admissions: HashMap<RequestKey, Admission>,
    admitted_pools: HashSet<(usize, RequestKey)>,
    operation_routes: HashMap<(RequestKey, uniserve_worker_ipc::OpId), usize>,
    product_routes: HashMap<ProductRef, ProductRoute>,
    transfer_products: HashMap<ProductRef, ProductPayload>,
    pool_submissions: Vec<VecDeque<PoolSubmission>>,
    pool_collective_seqs: Vec<Option<u64>>,
    observed_collective_seqs: BTreeSet<u64>,
    collective_frontier: u64,
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

impl StagedExecutor {
    /// Build a staged router after validating exact, non-overlapping pool claims.
    pub fn try_new(pools: Vec<(WorkerKind, Box<dyn Executor>)>) -> anyhow::Result<Self> {
        anyhow::ensure!(!pools.is_empty(), "StagedExecutor needs at least one pool");
        anyhow::ensure!(
            pools.len() <= u64::BITS as usize,
            "StagedExecutor supports at most {} pools",
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
        let pool_count = pools.len();
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
            pool_submissions: (0..pool_count).map(|_| VecDeque::new()).collect(),
            pool_collective_seqs: vec![None; pool_count],
            observed_collective_seqs: BTreeSet::new(),
            collective_frontier: 0,
            next_call_id: 1,
        })
    }

    fn merge_caps(
        pools: &[PoolEntry],
        routing: &HashMap<ForwardMode, usize>,
    ) -> anyhow::Result<WorkerCapabilities> {
        let routed_caps =
            |variant: ForwardMode| routing.get(&variant).map(|index| pools[*index].exec.caps());
        let mut kv_pool_indices = [
            ForwardMode::TokenExtend,
            ForwardMode::TokenDecode,
            ForwardMode::TokenVerify,
            ForwardMode::Materialize,
            ForwardMode::TransferKvPublish,
            ForwardMode::TransferKvInstall,
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
            .filter(|caps| !caps.model_identity.is_empty() || !caps.weight_digest.is_empty())
            .map(|caps| (caps.model_identity, caps.weight_digest))
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
            merged.model_identity = identity.0.clone();
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
                        && other.num_kv_heads == first.num_kv_heads
                        && other.head_dim == first.head_dim
                        && other.kv_dtype == first.kv_dtype
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
            merged.num_kv_heads = first.num_kv_heads;
            merged.head_dim = first.head_dim;
            merged.groups = first.groups;
            merged.kv_dtype = first.kv_dtype;
            merged.bytes_per_token = kv_pool_indices
                .iter()
                .map(|index| pools[*index].exec.caps().bytes_per_token)
                .max()
                .unwrap_or(first.bytes_per_token);
        }

        merged.supported_work = ForwardMode::ALL
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
        merged.max_batch_operations = pools
            .iter()
            .map(|pool| pool.exec.caps().max_batch_operations)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
        merged.max_batch_tokens = pools
            .iter()
            .map(|pool| pool.exec.caps().max_batch_tokens)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
        merged.max_request_pool_size = pools
            .iter()
            .map(|pool| pool.exec.caps().max_request_pool_size)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
        merged.max_unresolved_window = pools
            .iter()
            .map(|pool| pool.exec.caps().max_unresolved_window)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
        merged.incremental_kv_publication = pools
            .iter()
            .all(|pool| pool.exec.caps().incremental_kv_publication);
        let sampling_ownership = pools[0].exec.caps().sampling_ownership;
        anyhow::ensure!(
            pools
                .iter()
                .all(|pool| pool.exec.caps().sampling_ownership == sampling_ownership),
            "staged pools disagree on sampling ownership"
        );
        merged.sampling_ownership = sampling_ownership;
        merged.mixed_buckets = match (
            routing.get(&ForwardMode::TokenDecode),
            routing.get(&ForwardMode::GenFlow),
        ) {
            (Some(decode), Some(flow)) if decode == flow => {
                pools[*decode].exec.caps().mixed_buckets.clone()
            }
            _ => Vec::new(),
        };

        let flow = routed_caps(ForwardMode::GenFlow);
        merged.latent_page_units = flow.as_ref().map_or(0, |caps| caps.latent_page_units);
        merged.num_latent_pages = flow.as_ref().map_or(0, |caps| caps.num_latent_pages);
        merged.latent_width = flow.as_ref().map_or(0, |caps| caps.latent_width);
        merged.latent_dtype = flow
            .as_ref()
            .map_or_else(String::new, |caps| caps.latent_dtype.clone());
        merged.latent_downsample = flow.as_ref().map_or(0, |caps| caps.latent_downsample);
        merged.max_cfg_branches = flow.as_ref().map_or(0, |caps| caps.max_cfg_branches);
        merged.max_vae_grid_tokens = routed_caps(ForwardMode::EncodeLatent)
            .as_ref()
            .map_or(0, |caps| caps.max_vae_grid_tokens);
        merged.max_vit_grid_tokens = routed_caps(ForwardMode::EncodeVision)
            .as_ref()
            .map_or(0, |caps| caps.max_vit_grid_tokens);
        merged.max_latent_feature_bytes = routed_caps(ForwardMode::EncodeLatent)
            .as_ref()
            .map_or(0, |caps| caps.max_latent_feature_bytes);
        merged.max_vision_feature_bytes = routed_caps(ForwardMode::EncodeVision)
            .as_ref()
            .map_or(0, |caps| caps.max_vision_feature_bytes);
        if let Some(materialize) = routed_caps(ForwardMode::Materialize) {
            merged.commit_marker_tokens = materialize.commit_marker_tokens;
            merged.gen_rope_advance = materialize.gen_rope_advance;
        }
        merged.encoder_cache_budget = [ForwardMode::EncodeLatent, ForwardMode::EncodeVision]
            .into_iter()
            .filter_map(|variant| routed_caps(variant).map(|caps| caps.encoder_cache_budget))
            .min()
            .unwrap_or(0);
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
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            !partitions.is_empty() || !controls.is_empty(),
            "staged pool submission has no work or control"
        );
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
        Ok(())
    }

    fn dispatch_ready(&mut self) -> anyhow::Result<()> {
        loop {
            let mut progressed = false;
            for pool_index in 0..self.pools.len() {
                if !self.pools[pool_index].exec.can_submit() {
                    continue;
                }
                let ready = self.pool_submissions[pool_index]
                    .front()
                    .is_some_and(|submission| {
                        let sequence_ready =
                            submission.collective_seqs.first().is_none_or(|first| {
                                *first <= self.collective_frontier
                                    && self.pool_collective_seqs[pool_index]
                                        .is_none_or(|previous| *first > previous)
                            });
                        sequence_ready
                            && submission
                                .dependencies
                                .iter()
                                .all(|product| self.transfer_products.contains_key(product))
                    });
                if !ready {
                    continue;
                }
                let mut submission = self.pool_submissions[pool_index]
                    .pop_front()
                    .ok_or_else(|| anyhow::anyhow!("ready staged submission disappeared"))?;
                for dependency in submission.dependencies {
                    anyhow::ensure!(
                        !submission
                            .input_products
                            .iter()
                            .any(|payload| payload.product == dependency),
                        "cross-stage input payload is supplied more than once"
                    );
                    let payload = self.transfer_products.get(&dependency).ok_or_else(|| {
                        anyhow::anyhow!(
                            "ready cross-stage input {:?} lost its transfer descriptor",
                            dependency
                        )
                    })?;
                    submission.input_products.push(payload.clone());
                }
                self.submit_partition(
                    pool_index,
                    submission.step_id,
                    submission.partitions,
                    submission.controls,
                    submission.input_products,
                )?;
                if let Some(last) = submission.collective_seqs.last() {
                    self.pool_collective_seqs[pool_index] = Some(*last);
                }
                progressed = true;
            }
            if !progressed {
                return Ok(());
            }
        }
    }

    fn pump(&mut self) -> anyhow::Result<()> {
        self.dispatch_ready()?;
        for pool_index in 0..self.pools.len() {
            loop {
                let report = self.pools[pool_index].exec.poll()?;
                let Some(report) = report else {
                    break;
                };
                self.route_result(pool_index, report)?;
            }
        }
        self.dispatch_ready()
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
        self.dispatch_ready()?;
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

impl Executor for StagedExecutor {
    fn caps(&self) -> WorkerCapabilities {
        self.caps.clone()
    }

    fn pipeline_depth(&self) -> usize {
        self.depth
    }

    fn in_flight(&self) -> usize {
        self.pending.len() + self.ready.len()
    }

    fn device_products_reachable(&self, producer: ForwardMode, consumer: ForwardMode) -> bool {
        self.routing.contains_key(&producer) && self.routing.contains_key(&consumer)
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
        let mut pool_dependencies = (0..self.pools.len())
            .map(|_| Vec::new())
            .collect::<Vec<Vec<ProductRef>>>();
        let mut partition_routes = HashMap::with_capacity(batch.partitions.len());
        let mut input_routes = HashMap::new();
        let mut group_pools = HashMap::new();
        for partition in &batch.partitions {
            let mut pool_index = None;
            for operation in &partition.operations {
                let variant = operation.work;
                let operation_pool = self.routing.get(&variant).copied().ok_or_else(|| {
                    anyhow::anyhow!(
                        "StagedExecutor has no pool for work variant {variant:?} in request {:?}",
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
                if variant == ForwardMode::GenFlow {
                    let token_pool = self
                        .routing
                        .get(&ForwardMode::TokenDecode)
                        .or_else(|| self.routing.get(&ForwardMode::TokenExtend));
                    if token_pool.is_some_and(|token_pool| *token_pool != operation_pool) {
                        anyhow::ensure!(
                            !operation.inputs.is_empty(),
                            "cross-pool generation flow for request {:?} names no input product for its conditioning",
                            operation.request_key
                        );
                    }
                }
                for input in
                    operation
                        .inputs
                        .iter()
                        .chain(operation.predicate.iter().filter(|predicate| {
                            !operation.inputs.iter().any(|input| input == *predicate)
                        }))
                {
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
                    "StagedExecutor cannot route undeclared input product {:?}",
                    payload.product
                )
            })?;
            partition_inputs[pool_index].push(payload.clone());
        }
        let mut cross_pool_consumers = HashSet::new();
        for partition in &batch.partitions {
            let consumer_pool = partition_routes[&partition.partition_id];
            for operation in &partition.operations {
                for input in
                    operation
                        .inputs
                        .iter()
                        .chain(operation.predicate.iter().filter(|predicate| {
                            !operation.inputs.iter().any(|input| input == *predicate)
                        }))
                {
                    if input.storage_class == uniserve_worker_ipc::StorageClass::HostStaging {
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
                    anyhow::ensure!(
                        !partition_inputs[consumer_pool]
                            .iter()
                            .any(|existing| existing.product == *input),
                        "cross-stage transfer descriptor must be supplied by its producing pool"
                    );
                    if !pool_dependencies[consumer_pool].contains(input) {
                        pool_dependencies[consumer_pool].push(input.clone());
                    }
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
        for (pool_index, (((partitions, controls), input_products), dependencies)) in
            pool_partitions
                .into_iter()
                .zip(pool_controls)
                .zip(partition_inputs)
                .zip(pool_dependencies)
                .enumerate()
        {
            if partitions.is_empty() && controls.is_empty() {
                anyhow::ensure!(
                    dependencies.is_empty(),
                    "cross-stage dependency has no consuming pool submission"
                );
                continue;
            }
            expected_pools |= Self::pool_bit(pool_index);
            let mut collective_seqs = partitions
                .iter()
                .map(|partition| partition.collective_seq)
                .collect::<Vec<_>>();
            collective_seqs.sort_unstable();
            collective_seqs.dedup();
            let submission = PoolSubmission {
                step_id: batch.step_id,
                partitions,
                controls,
                input_products,
                dependencies,
                collective_seqs,
            };
            let sequence = submission.collective_seqs.first().copied();
            self.observed_collective_seqs
                .extend(submission.collective_seqs.iter().copied());
            while self
                .observed_collective_seqs
                .remove(&self.collective_frontier.saturating_add(1))
            {
                self.collective_frontier = self.collective_frontier.saturating_add(1);
            }
            let position = self.pool_submissions[pool_index]
                .iter()
                .position(
                    |queued| match (sequence, queued.collective_seqs.first().copied()) {
                        (Some(incoming), Some(existing)) => incoming < existing,
                        (None, Some(_)) => true,
                        _ => false,
                    },
                )
                .unwrap_or(self.pool_submissions[pool_index].len());
            self.pool_submissions[pool_index].insert(position, submission);
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
        self.dispatch_ready()?;
        self.try_complete(batch.step_id)?;
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

    fn command_waker(&self) -> CommandWaker {
        let wakes = self
            .pools
            .iter()
            .map(|pool| pool.exec.command_waker())
            .collect::<Vec<_>>();
        CommandWaker::new(move || {
            for wake in &wakes {
                wake.wake();
            }
        })
    }

    fn wake_file_descriptors(&self) -> Vec<i32> {
        self.pools
            .iter()
            .flat_map(|pool| pool.exec.wake_file_descriptors())
            .collect()
    }

    fn park_for_event(&mut self, timeout: Duration) -> anyhow::Result<()> {
        crate::worker::park_descriptors(&self.wake_file_descriptors(), timeout)
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
            self.park_for_event(Duration::from_secs(300))?;
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
