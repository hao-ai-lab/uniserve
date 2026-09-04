//! Typed operation routing across heterogeneous worker pools.
//!
//! A topology routes operations by [`OpKind`] and admits a request to each pool
//! before its first operation there. Device products remain resident in their
//! producing pool. Cross-pool consumers wait for a bounded transfer descriptor,
//! then launch before the producer completion becomes scheduler-visible.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::time::Duration;

use crate::executor::{
    Batch, BatchResult, Executor, ExecutorInfo, ExecutorSubmitError, LogicalResultTracker, Op,
    PhysicalExecutor, PhysicalSubmitError, PoolConfig, physical_run,
};
use anyhow::Context;
use uniserve_core::{CommandWaker, RequestId};
use uniserve_worker_ipc::{
    BatchCommand, BufferId, BufferPlacement, InlineValue, NewRequest, OpKind, Operation,
    ProductPayload, ProductRef, RequestKey, Run as PhysicalRun, RunResult, TransferKind,
};

struct PoolEntry {
    config: PoolConfig,
    exec: Box<dyn PhysicalExecutor>,
}

struct PendingStep {
    expected_pools: u64,
    operations: HashMap<(RequestKey, uniserve_worker_ipc::OpId), usize>,
    returned_operations: HashSet<(RequestKey, uniserve_worker_ipc::OpId)>,
    finished_requests: Vec<RequestId>,
}

#[derive(Clone)]
struct PoolSubmission {
    run: PhysicalRun,
    dependencies: Vec<ProductRef>,
}

#[derive(Clone)]
struct ProductRoute {
    pool_index: usize,
}

/// Routes a canonical typed batch across the pools of a staged topology.
pub struct StagedExecutor {
    routing: HashMap<OpKind, usize>,
    pools: Vec<PoolEntry>,
    executor_info: ExecutorInfo,
    depth: usize,
    command_wakers: Vec<CommandWaker>,
    progress_fds: Vec<i32>,
    pending: BTreeMap<u64, PendingStep>,
    ready: VecDeque<RunResult>,
    admissions: HashMap<RequestKey, NewRequest>,
    admitted_pools: HashSet<(usize, RequestKey)>,
    operation_routes: HashMap<(RequestKey, uniserve_worker_ipc::OpId), usize>,
    product_routes: HashMap<ProductRef, ProductRoute>,
    buffer_placements: HashMap<BufferId, BufferPlacement>,
    buffer_pools: HashMap<BufferId, HashSet<usize>>,
    transfer_products: HashMap<ProductRef, ProductPayload>,
    pool_submissions: Vec<VecDeque<PoolSubmission>>,
    pool_collective_seqs: Vec<Option<u64>>,
    observed_collective_seqs: BTreeSet<u64>,
    collective_frontier: u64,
    next_collective_seq: u64,
    logical_results: LogicalResultTracker,
    command_wake_pending: bool,
}

impl StagedExecutor {
    /// Builds a staged router after validating exact, non-overlapping pool claims.
    pub fn try_new(pools: Vec<(PoolConfig, Box<dyn PhysicalExecutor>)>) -> anyhow::Result<Self> {
        Self::try_new_with_signals(pools, Vec::new(), Vec::new())
    }

    /// Builds a staged router with explicit command wakes and progress descriptors.
    pub(crate) fn try_new_with_signals(
        pools: Vec<(PoolConfig, Box<dyn PhysicalExecutor>)>,
        command_wakers: Vec<CommandWaker>,
        progress_fds: Vec<i32>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(!pools.is_empty(), "StagedExecutor needs at least one pool");
        anyhow::ensure!(
            pools.len() <= u64::BITS as usize,
            "StagedExecutor supports at most {} pools",
            u64::BITS
        );

        let mut routing = HashMap::new();
        let mut pool_ids = HashSet::new();
        for (index, (config, executor)) in pools.iter().enumerate() {
            anyhow::ensure!(
                pool_ids.insert(config.id.clone()),
                "duplicate staged pool id {}",
                config.id
            );
            let info = executor.physical_info().single_pool();
            info.validate()
                .with_context(|| format!("staged pool {index} reported invalid worker info"))?;
            anyhow::ensure!(
                config.queue_depth.max(1) == info.queue_depth.max(1) as usize,
                "staged pool {index} configured depth {} disagrees with worker depth {}",
                config.queue_depth.max(1),
                info.queue_depth.max(1)
            );
            for variant in &config.supported_ops {
                anyhow::ensure!(
                    info.supported_ops.contains(variant),
                    "staged pool {} does not report configured operation {variant:?}",
                    config.id
                );
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
            .map(|(config, exec)| PoolEntry { config, exec })
            .collect();
        let executor_info = ExecutorInfo::from_pools(
            pools
                .iter()
                .map(|pool| {
                    (
                        pool.config.id.clone(),
                        pool.exec.physical_info().single_pool().clone(),
                    )
                })
                .collect(),
        )?;
        executor_info.runtime_info()?;
        let depth = pools
            .iter()
            .map(|pool| pool.config.queue_depth.max(1))
            .min()
            .unwrap_or(1)
            .max(1);
        let pool_count = pools.len();
        Ok(Self {
            routing,
            pools,
            executor_info,
            depth,
            command_wakers,
            progress_fds,
            pending: BTreeMap::new(),
            ready: VecDeque::new(),
            admissions: HashMap::new(),
            admitted_pools: HashSet::new(),
            operation_routes: HashMap::new(),
            product_routes: HashMap::new(),
            buffer_placements: HashMap::new(),
            buffer_pools: HashMap::new(),
            transfer_products: HashMap::new(),
            pool_submissions: (0..pool_count).map(|_| VecDeque::new()).collect(),
            pool_collective_seqs: vec![None; pool_count],
            observed_collective_seqs: BTreeSet::new(),
            collective_frontier: 0,
            next_collective_seq: 1,
            logical_results: LogicalResultTracker::default(),
            command_wake_pending: false,
        })
    }

    /// Returns the mask bit assigned to a command pool.
    fn pool_bit(index: usize) -> u64 {
        1u64 << index
    }

    /// Returns the command pools eligible for cache admission.
    fn cache_admissions(&mut self, admissions: &[NewRequest]) -> anyhow::Result<()> {
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

    /// Partitions new-request admissions by the pools used by their first operations.
    fn admissions_for(
        &self,
        pool_index: usize,
        operations: &[Operation],
    ) -> anyhow::Result<Vec<NewRequest>> {
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

    /// Returns the pools that must receive a control command. A commit or close rides with
    /// the operation that produced the version it names, so it targets the pool
    /// running this request's operation in this batch; a control for a request
    /// with no operation this batch is broadcast to every pool that has admitted
    /// the request.
    fn command_pools(&self, command: &BatchCommand) -> Vec<usize> {
        if let BatchCommand::Free { buffer } = command {
            return self
                .buffer_pools
                .get(buffer)
                .map(|pools| {
                    let mut pools = pools.iter().copied().collect::<Vec<_>>();
                    pools.sort_unstable();
                    pools
                })
                .unwrap_or_default();
        }
        let (request_key, producer_op_id) = match command {
            BatchCommand::Start { .. } => return Vec::new(),
            BatchCommand::Commit {
                request_key,
                selected,
                ..
            } => (*request_key, selected.op_id),
            BatchCommand::Finish {
                request_key,
                cutoff,
                ..
            } => (*request_key, cutoff.op_id),
            BatchCommand::Free { .. } => unreachable!("buffer free was routed above"),
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

    /// Submits or queues one pool-local run while preserving collective order.
    fn submit_pool(
        &mut self,
        pool_index: usize,
        submission: &PoolSubmission,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(
            !submission.run.operations.is_empty() || !submission.run.commands.is_empty(),
            "staged pool submission has no work or control"
        );
        let admissions = self.admissions_for(pool_index, &submission.run.operations)?;
        let request_keys = admissions
            .iter()
            .map(|admission| admission.request_key)
            .collect::<Vec<_>>();
        let mut run = submission.run.clone();
        let mut commands = admissions
            .into_iter()
            .map(|request| BatchCommand::Start { request })
            .collect::<Vec<_>>();
        commands.append(&mut run.commands);
        run.commands = commands;
        run.validate()?;
        match self.pools[pool_index].exec.submit_run(run) {
            Ok(()) => {}
            Err(PhysicalSubmitError::WouldBlock(_)) => return Ok(false),
            Err(PhysicalSubmitError::Failed(error)) => return Err(error),
        }
        self.admitted_pools
            .extend(request_keys.into_iter().map(|key| (pool_index, key)));
        Ok(true)
    }

    /// Dispatches queued pool runs whose collective and capacity constraints are satisfied.
    fn dispatch_ready(&mut self) -> anyhow::Result<()> {
        loop {
            let mut progressed = false;
            for pool_index in 0..self.pools.len() {
                let ready = self.pool_submissions[pool_index]
                    .front()
                    .is_some_and(|submission| {
                        let sequence_ready = {
                            let sequence = submission.run.collective_seq;
                            sequence <= self.collective_frontier
                                && self.pool_collective_seqs[pool_index]
                                    .is_none_or(|previous| sequence > previous)
                        };
                        sequence_ready
                            && submission
                                .dependencies
                                .iter()
                                .all(|product| self.transfer_products.contains_key(product))
                    });
                if !ready {
                    continue;
                }
                let submission = self.pool_submissions[pool_index]
                    .pop_front()
                    .ok_or_else(|| anyhow::anyhow!("ready staged submission disappeared"))?;
                let mut physical_submission = submission.clone();
                for dependency in &submission.dependencies {
                    anyhow::ensure!(
                        !physical_submission
                            .run
                            .input_products
                            .iter()
                            .any(|payload| payload.product == *dependency),
                        "cross-stage input payload is supplied more than once"
                    );
                    let payload = self.transfer_products.get(&dependency).ok_or_else(|| {
                        anyhow::anyhow!(
                            "ready cross-stage input {:?} lost its transfer descriptor",
                            dependency
                        )
                    })?;
                    physical_submission.run.input_products.push(payload.clone());
                }
                if !self.submit_pool(pool_index, &physical_submission)? {
                    self.pool_submissions[pool_index].push_front(submission);
                    continue;
                }
                self.pool_collective_seqs[pool_index] = Some(submission.run.collective_seq);
                progressed = true;
            }
            if !progressed {
                return Ok(());
            }
        }
    }

    /// Advances staged command execution until no immediate progress remains.
    fn pump(&mut self) -> anyhow::Result<()> {
        self.dispatch_ready()?;
        for pool_index in 0..self.pools.len() {
            loop {
                let report = self.pools[pool_index].exec.poll_run(Duration::ZERO)?;
                if self.pools[pool_index].exec.take_command_wake() {
                    self.command_wake_pending = true;
                }
                let Some(report) = report else {
                    break;
                };
                self.route_result(pool_index, report)?;
            }
        }
        self.dispatch_ready()
    }

    /// Routes a pool result into its staged aggregate and publishes transferable products.
    fn route_result(&mut self, pool_index: usize, report: RunResult) -> anyhow::Result<()> {
        report.validate()?;
        let mut report = report;
        let run_id = report.run_id;
        let pool_bit = Self::pool_bit(pool_index);
        {
            let step = self
                .pending
                .get_mut(&run_id)
                .ok_or_else(|| anyhow::anyhow!("stage pool returned unknown step {run_id}"))?;
            anyhow::ensure!(
                step.expected_pools & pool_bit != 0,
                "pool {pool_index} returned a duplicate or unexpected result for step {run_id}"
            );
            let expected_operations = step
                .operations
                .iter()
                .filter_map(|(identity, route)| (*route == pool_index).then_some(*identity))
                .collect::<HashSet<_>>();
            anyhow::ensure!(
                !report.completions.is_empty() || expected_operations.is_empty() || report.done,
                "pool {pool_index} returned an empty partial report for step {run_id}"
            );
            for completion in &report.completions {
                let identity = (completion.request_key, completion.op_id);
                anyhow::ensure!(
                    expected_operations.contains(&identity),
                    "pool {pool_index} returned an unexpected operation for step {run_id}"
                );
                anyhow::ensure!(
                    step.returned_operations.insert(identity),
                    "pool {pool_index} returned an operation twice for step {run_id}"
                );
            }
            if report.done {
                anyhow::ensure!(
                    expected_operations
                        .iter()
                        .all(|identity| step.returned_operations.contains(identity)),
                    "pool {pool_index} terminated step {run_id} before every operation completed"
                );
                step.expected_pools &= !pool_bit;
            }
        }
        let mut visible_products = Vec::with_capacity(report.products.len());
        for product in report.products.drain(..) {
            let InlineValue::Transfer(handle) = &product.value else {
                visible_products.push(product);
                continue;
            };
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
            let kind = handle.kind();
            if let Some(existing) = self.transfer_products.get(&product.product) {
                anyhow::ensure!(
                    existing == &product,
                    "cross-stage product identity was reused with conflicting transport bytes"
                );
            } else {
                self.transfer_products
                    .insert(product.product.clone(), product.clone());
            }
            if kind == TransferKind::Kv {
                visible_products.push(ProductPayload {
                    product: product.product,
                    value: InlineValue::Bytes(Vec::new()),
                });
            }
        }
        report.products = visible_products;
        self.dispatch_ready()?;
        let done = self.try_complete(run_id)?;
        report.done = done;
        if !report.completions.is_empty() || done {
            self.ready.push_back(report);
        }
        Ok(())
    }

    /// Finalizes a staged run once every required pool has returned its contribution.
    fn try_complete(&mut self, run_id: u64) -> anyhow::Result<bool> {
        let complete = self
            .pending
            .get(&run_id)
            .is_some_and(|step| step.expected_pools == 0);
        if !complete {
            return Ok(false);
        }
        let step = self
            .pending
            .remove(&run_id)
            .ok_or_else(|| anyhow::anyhow!("pending step {run_id} disappeared"))?;
        anyhow::ensure!(
            step.returned_operations.len() == step.operations.len(),
            "stage execution finished without every planned operation"
        );
        for request_id in step.finished_requests {
            self.forget_request(request_id);
        }
        Ok(true)
    }

    /// Forgets the request.
    fn forget_request(&mut self, request_id: RequestId) {
        self.admissions
            .retain(|request_key, _| request_key.request_id != request_id);
        self.admitted_pools
            .retain(|(_, request_key)| request_key.request_id != request_id);
        self.operation_routes
            .retain(|(request_key, _), _| request_key.request_id != request_id);
        self.product_routes.retain(|product, _| {
            product.request_key.request_id != request_id || product.uses_persistent_buffer()
        });
        self.transfer_products.retain(|product, _| {
            product.request_key.request_id != request_id || product.uses_persistent_buffer()
        });
    }

    /// Returns the command-channel waker.
    pub(crate) fn command_waker(&self) -> CommandWaker {
        let wakes = self.command_wakers.clone();
        CommandWaker::new(move || {
            for wake in &wakes {
                wake.wake();
            }
        })
    }
}

impl Executor for StagedExecutor {
    /// Returns the worker metadata.
    fn info(&self) -> &ExecutorInfo {
        &self.executor_info
    }

    /// Partitions a logical batch across owning pools and registers aggregate completion state.
    fn submit(&mut self, batch: Batch) -> Result<(), ExecutorSubmitError> {
        self.pump().map_err(ExecutorSubmitError::Failed)?;
        if self.pending.len() >= self.depth {
            return Err(ExecutorSubmitError::WouldBlock(batch));
        }
        batch.validate().map_err(ExecutorSubmitError::Failed)?;
        let batch_id = batch.id;
        if self.pending.contains_key(&batch_id) {
            return Err(ExecutorSubmitError::Failed(anyhow::anyhow!(
                "stage router already has batch {batch_id} in flight"
            )));
        }
        self.logical_results
            .register(&batch)
            .map_err(ExecutorSubmitError::Failed)?;
        let admissions = batch.admissions().cloned().collect::<Vec<_>>();
        self.cache_admissions(&admissions)
            .map_err(ExecutorSubmitError::Failed)?;

        let submit_result = (|| -> anyhow::Result<()> {
            let mut pool_ops = (0..self.pools.len())
                .map(|_| Vec::new())
                .collect::<Vec<Vec<Op>>>();
            let mut pool_inputs = (0..self.pools.len())
                .map(|_| Vec::new())
                .collect::<Vec<Vec<ProductPayload>>>();
            let mut pool_dependencies = (0..self.pools.len())
                .map(|_| Vec::new())
                .collect::<Vec<Vec<ProductRef>>>();
            let mut operation_routes = HashMap::with_capacity(batch.ops.len());
            let mut input_routes = HashMap::new();
            for op in &batch.ops {
                let operation = op.clone().into_operation();
                let variant = op.kind;
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
                for output in operation.outputs() {
                    let route = ProductRoute {
                        pool_index: operation_pool,
                    };
                    if let Some(existing) = self.product_routes.get(output) {
                        anyhow::ensure!(
                            existing.pool_index == route.pool_index,
                            "product identity was routed to conflicting pools"
                        );
                    } else {
                        self.product_routes.insert(output.clone(), route);
                    }
                    if output.uses_persistent_buffer() {
                        let placement = op
                            .placement
                            .buffers
                            .iter()
                            .find(|placement| placement.buffer == output.buffer_id())
                            .copied()
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "persistent product {:?} has no scheduler placement",
                                    output
                                )
                            })?;
                        if let Some(existing) =
                            self.buffer_placements.insert(placement.buffer, placement)
                        {
                            anyhow::ensure!(
                                existing == placement,
                                "buffer identity was assigned conflicting placements"
                            );
                        }
                        self.buffer_pools
                            .entry(placement.buffer)
                            .or_default()
                            .insert(operation_pool);
                    }
                }
                operation_routes.insert((operation.request_key, operation.op_id), operation_pool);
                if variant == OpKind::DiffusionStep {
                    let token_pool = self
                        .routing
                        .get(&OpKind::ArDecode)
                        .or_else(|| self.routing.get(&OpKind::ArExtend));
                    if token_pool.is_some_and(|token_pool| *token_pool != operation_pool) {
                        anyhow::ensure!(
                            !operation.inputs().is_empty(),
                            "cross-pool media denoise for request {:?} names no input product for its conditioning",
                            operation.request_key
                        );
                    }
                }
                for input in operation.inputs().iter().chain(
                    operation
                        .predicate()
                        .into_iter()
                        .filter(|predicate| !operation.inputs().contains(*predicate)),
                ) {
                    anyhow::ensure!(
                        input_routes
                            .insert(input.clone(), operation_pool)
                            .is_none_or(|existing| existing == operation_pool),
                        "one input product is consumed across staged pools"
                    );
                }
                pool_ops[operation_pool].push(op.clone());
            }
            for payload in &batch.inline {
                let pool_index = input_routes.get(&payload.product).copied().ok_or_else(|| {
                    anyhow::anyhow!(
                        "StagedExecutor cannot route undeclared input product {:?}",
                        payload.product
                    )
                })?;
                pool_inputs[pool_index].push(payload.clone());
            }
            for op in &batch.ops {
                let operation = op.clone().into_operation();
                let consumer_pool = operation_routes[&(operation.request_key, operation.op_id)];
                for input in operation.inputs().iter().chain(
                    operation
                        .predicate()
                        .into_iter()
                        .filter(|predicate| !operation.inputs().contains(*predicate)),
                ) {
                    if input.storage_class == uniserve_worker_ipc::StorageClass::HostStaging {
                        continue;
                    }
                    let producer = self.product_routes.get(input).ok_or_else(|| {
                        anyhow::anyhow!("cross-stage input {:?} has no registered producer", input)
                    })?;
                    if producer.pool_index == consumer_pool {
                        continue;
                    }
                    anyhow::ensure!(
                        !pool_inputs[consumer_pool]
                            .iter()
                            .any(|existing| existing.product == *input),
                        "cross-stage transfer descriptor must be supplied by its producing pool"
                    );
                    if !pool_dependencies[consumer_pool].contains(input) {
                        pool_dependencies[consumer_pool].push(input.clone());
                    }
                    if input.uses_persistent_buffer() {
                        self.buffer_pools
                            .entry(input.buffer_id())
                            .or_default()
                            .insert(consumer_pool);
                    }
                }
            }

            let mut pool_commands = (0..self.pools.len())
                .map(|_| Vec::new())
                .collect::<Vec<Vec<BatchCommand>>>();
            for command in &batch.commands {
                if matches!(command, BatchCommand::Start { .. }) {
                    continue;
                }
                let request_key = command.request_key();
                let targets = self.command_pools(command);
                anyhow::ensure!(
                    !targets.is_empty(),
                    "stage router received a command for request {request_key:?} that is not admitted to any pool"
                );
                for pool_index in targets {
                    pool_commands[pool_index].push(command.clone());
                }
            }

            let mut expected_pools = 0;
            for (pool_index, (((ops, commands), input_products), dependencies)) in pool_ops
                .into_iter()
                .zip(pool_commands)
                .zip(pool_inputs)
                .zip(pool_dependencies)
                .enumerate()
            {
                if ops.is_empty() && commands.is_empty() {
                    anyhow::ensure!(
                        dependencies.is_empty(),
                        "cross-stage dependency has no consuming pool submission"
                    );
                    continue;
                }
                expected_pools |= Self::pool_bit(pool_index);
                let collective_seq = self.next_collective_seq.max(1);
                self.next_collective_seq = collective_seq.saturating_add(1);
                let mut run = physical_run(
                    batch_id,
                    batch_id,
                    collective_seq,
                    ops,
                    commands,
                    input_products,
                )?;
                for dependency in &dependencies {
                    if !dependency.uses_persistent_buffer() {
                        continue;
                    }
                    let placement = self
                        .buffer_placements
                        .get(&dependency.buffer_id())
                        .copied()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "persistent dependency {:?} has no scheduler placement",
                                dependency
                            )
                        })?;
                    if !run
                        .buffer_placements
                        .iter()
                        .any(|existing| existing.buffer == placement.buffer)
                    {
                        run.buffer_placements.push(placement);
                    }
                }
                run.validate()?;
                let submission = PoolSubmission { run, dependencies };
                self.observed_collective_seqs.insert(collective_seq);
                while self
                    .observed_collective_seqs
                    .remove(&self.collective_frontier.saturating_add(1))
                {
                    self.collective_frontier = self.collective_frontier.saturating_add(1);
                }
                let position = self.pool_submissions[pool_index]
                    .iter()
                    .position(|queued| collective_seq < queued.run.collective_seq)
                    .unwrap_or(self.pool_submissions[pool_index].len());
                self.pool_submissions[pool_index].insert(position, submission);
            }
            self.pending.insert(
                batch_id,
                PendingStep {
                    expected_pools,
                    operations: operation_routes,
                    returned_operations: HashSet::new(),
                    finished_requests: batch
                        .commands
                        .iter()
                        .filter_map(|control| match control {
                            BatchCommand::Finish { request_key, .. } => {
                                Some(request_key.request_id)
                            }
                            _ => None,
                        })
                        .collect(),
                },
            );
            self.dispatch_ready()?;
            self.try_complete(batch_id)?;
            for command in &batch.commands {
                match command {
                    BatchCommand::Free { buffer } => {
                        self.transfer_products
                            .retain(|product, _| product.buffer_id() != *buffer);
                        self.product_routes
                            .retain(|product, _| product.buffer_id() != *buffer);
                        self.buffer_placements.remove(buffer);
                        self.buffer_pools.remove(buffer);
                    }
                    _ => {}
                }
            }
            Ok(())
        })();
        if submit_result.is_err() {
            self.logical_results.unregister(batch_id);
        }
        submit_result.map_err(ExecutorSubmitError::Failed)
    }

    /// Drives pool progress and returns the next completed logical batch.
    fn poll(&mut self, timeout: Duration) -> anyhow::Result<Option<BatchResult>> {
        if std::mem::take(&mut self.command_wake_pending) {
            return Ok(None);
        }
        self.pump()?;
        if std::mem::take(&mut self.command_wake_pending) {
            return Ok(None);
        }
        if let Some(result) = self.ready.pop_front() {
            return self.logical_results.apply(result).map(Some);
        }
        if timeout.is_zero() {
            return Ok(None);
        }
        if self.progress_fds.is_empty() {
            let divisor = u32::try_from(self.pools.len()).unwrap_or(1).max(1);
            let per_pool = timeout / divisor;
            for pool_index in 0..self.pools.len() {
                let report = self.pools[pool_index].exec.poll_run(per_pool)?;
                if self.pools[pool_index].exec.take_command_wake() {
                    return Ok(None);
                }
                if let Some(report) = report {
                    self.route_result(pool_index, report)?;
                    if let Some(result) = self.ready.pop_front() {
                        return self.logical_results.apply(result).map(Some);
                    }
                }
            }
            return Ok(None);
        }
        crate::worker::park_descriptors(&self.progress_fds, timeout)?;
        self.pump()?;
        if std::mem::take(&mut self.command_wake_pending) {
            return Ok(None);
        }
        self.ready
            .pop_front()
            .map(|result| self.logical_results.apply(result))
            .transpose()
    }

    /// Closes the component and releases its resources.
    fn close(&mut self) -> anyhow::Result<()> {
        let mut first_error = None;
        for pool in &mut self.pools {
            if let Err(error) = pool.exec.close_physical()
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        if let Some(error) = first_error {
            Err(error)
        } else {
            Ok(())
        }
    }
}
