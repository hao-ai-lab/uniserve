//! Physical worker fan-out, completion agreement, and process recovery.
//!
//! A `WorkerGroup` presents one worker's cooperating ranks to `WorkerExecutor`
//! as a single submission queue of `depth` physical runs. `submit_batch` uses
//! `rank_projection` to cut each `Batch` into per-rank projections under the
//! same batch identifier, each holding the calls that rank's components
//! execute and every command of the batch, and sends each projection through
//! that rank's `RankProcess`. Returned rank results wait in per-rank buffers
//! until every rank planned for a set of calls has reported them; `try_join`
//! then merges those reports into one `WorkerResult` published by the calls'
//! output owner, the first rank of their component. An execution error that
//! every failing rank reports identically, while no succeeding rank holds an
//! unjoined call, retires only its batch unless it is fatal, is coded as a
//! scheduler bug or invariant violation, or failed a batch carrying a `Free`
//! command (`join_rank_errors`); rank loss, protocol violations, and every
//! other error replace the whole rank group (`recover_workers`) and
//! invalidate every request resident on it.
//!
//! The module also launches groups: `spawn_all` starts every group's ranks,
//! locally or through the host launchers in `launcher`, before any group
//! waits for its ranks' endpoint reports and startup capabilities.

use crate::executor::{CallResult, WorkerResult};
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use super::registration::RankRegistry;
use super::{BatchSubmitError, PendingRank, RankProcess};
use crate::executor::{WorkerExecError, WorkerFailure};
use anyhow::Context;
use uniserve_worker_ipc::{Batch, BatchCommand, RequestKey, WorkerInfo};

use crate::worker::WorkerProcessArgs;

/// Maximum interval without rank progress before an in-flight batch is treated as failed.
///
/// Measured from `WorkerGroup::last_progress` and enforced only while a batch
/// is pending. Expiry is an error that `poll_batch` answers by replacing the
/// rank group.
const NEXT_RESULT_DEADLINE: Duration = Duration::from_secs(300);

/// Every rank of one group, with the launchers that started the remote ones.
type LaunchedGroup = (Vec<RankProcess>, Option<super::launcher::Launchers>);

/// One worker group's spawned ranks and the registry they report their
/// endpoints to.
pub(crate) struct LaunchedRanks {
    registry: RankRegistry,
    ranks: Vec<PendingRank>,
    /// Retained so this instance's launchers stay connected; closing a
    /// launcher connection terminates the ranks it started.
    launchers: Option<super::launcher::Launchers>,
}

impl WorkerProcessArgs {
    /// Launches and connects every rank in one physical worker group.
    ///
    /// `launchers` is the deployment's registry, which owns the launcher of
    /// every host this process does not run on; a group placing a rank on
    /// such a host requires it.
    fn launch(
        &self,
        cancel: Option<Arc<AtomicBool>>,
        launchers: Option<super::launcher::Launchers>,
    ) -> anyhow::Result<LaunchedGroup> {
        self.adopt_ranks(self.spawn_ranks(cancel, launchers)?)
    }

    /// Starts every rank of one group without waiting for its endpoint report.
    ///
    /// Spawning is separated from adoption so several groups start their
    /// processes concurrently and then wait for all of their reports together.
    fn spawn_ranks(
        &self,
        cancel: Option<Arc<AtomicBool>>,
        launchers: Option<super::launcher::Launchers>,
    ) -> anyhow::Result<LaunchedRanks> {
        let cancel = cancel.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        crate::WorkerConfig::validate_members(&self.ranks, &self.components)?;
        // This process spawns exactly the ranks placed on its own host. A rank
        // placed elsewhere is started by that host's launcher, which presented
        // to the deployment's registry before any rank is sent to it.
        let places_remotely = self.ranks.iter().any(|rank| rank.node != self.host);
        let launchers = match launchers {
            Some(registry) if places_remotely => Some(registry),
            None if places_remotely => anyhow::bail!(
                "worker {} places a rank on another host but no launcher registry was bound",
                self.worker_id
            ),
            _ => None,
        };

        // A group of several ranks rendezvouses at one TCP address for its
        // entire lifetime, and every rank reports its endpoint to one
        // registration address. Both are loopback where every rank runs here;
        // where some rank runs elsewhere both name a routable host, because a
        // loopback address reaches only the host that binds it. The first
        // rank serves the rendezvous store, so the store is placed on that
        // rank's host: this one, or the host whose launcher reserved a port
        // for it. Whichever process spawns the first rank holds the bound
        // socket until that rank inherits it, so the port is never free while
        // the rank starts.
        let head = match launchers.as_ref() {
            Some(registry) => Some(super::launcher::lock(registry)?.reachable_host()?),
            None => None,
        };
        let mut rendezvous = if self.ranks.len() > 1 {
            let first = &self.ranks[0].node;
            Some(match launchers.as_ref() {
                Some(registry) if *first != self.host => {
                    super::launcher::lock(registry)?.rendezvous_on(first, &self.worker_id)?
                }
                _ => super::registration::reserve_rendezvous(head)?,
            })
        } else {
            None
        };

        // Ranks name their own channel endpoints and report them here; the
        // engine binds each channel from the report rather than choosing the
        // endpoint before the process exists.
        let registry = RankRegistry::bind(head)?;
        let mut ranks = Vec::with_capacity(self.ranks.len());
        for rank in 0..self.ranks.len() {
            // A rank placed elsewhere is delivered to the launcher that owns
            // its host; a rank placed here is spawned by this process.
            let mut remote = match (&launchers, self.ranks[rank].node == self.host) {
                (Some(registry), false) => Some(super::launcher::RemoteHost {
                    registry: registry.as_ref(),
                    host: &self.ranks[rank].node,
                }),
                _ => None,
            };
            // Only the first rank serves the store; the others are its clients
            // and bind nothing, so they receive only its address.
            let store_listener = match rendezvous.as_mut() {
                Some(rendezvous) if rank == 0 => rendezvous.listener.take(),
                _ => None,
            };
            ranks.push(PendingRank::spawn_rank(
                self,
                rank as u32,
                rendezvous
                    .as_ref()
                    .map(|rendezvous| rendezvous.address.as_str()),
                store_listener,
                remote.as_mut(),
                Arc::clone(&cancel),
                registry.address(),
            )?);
        }

        Ok(LaunchedRanks {
            registry,
            ranks,
            launchers,
        })
    }

    /// Binds every rank's channel to the endpoint that rank reported.
    fn adopt_ranks(&self, launched: LaunchedRanks) -> anyhow::Result<LaunchedGroup> {
        let LaunchedRanks {
            registry,
            mut ranks,
            mut launchers,
        } = launched;
        let reports = registry.collect(&self.worker_id, ranks.len(), || {
            // A rank this process spawned is checked here; a rank another
            // host started reaches the head as an exit report from the
            // launcher that owns it, which fails the launch by name rather
            // than consuming the registration deadline.
            if let Some(exits) = launchers
                .as_ref()
                .map(super::launcher::lock)
                .transpose()?
                .map(|mut hosts| hosts.drain_exits(&self.worker_id))
                && let Some((host, exit)) = exits.into_iter().next()
            {
                anyhow::bail!(
                    "rank {} on host {host} exited with {} before reporting its endpoint",
                    exit.rank,
                    exit.status
                );
            }
            ranks
                .iter_mut()
                .try_for_each(|pending| pending.check_alive())
        })?;
        ranks
            .into_iter()
            .zip(reports)
            .map(|(pending, report)| {
                anyhow::ensure!(
                    pending.rank() == report.rank,
                    "rank {} adopted the endpoint reported by rank {}",
                    pending.rank(),
                    report.rank
                );
                pending.adopt(&report.transport, &report.endpoint)
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .map(|ranks| (ranks, launchers.take()))
    }
}

/// Cooperative worker processes with one IPC channel per rank.
pub struct WorkerGroup {
    /// Rank processes in rank order; empty after `close` or a failed recovery.
    workers: Vec<RankProcess>,
    /// Results each rank returned, indexed by rank. `take_rank_calls`
    /// consumes their calls as they are joined, and `finish_batch` drops what
    /// remains of a retired batch.
    buffers: Vec<VecDeque<WorkerResult>>,
    /// The group's agreed capabilities: rank 0's report with the group-wide
    /// product storage and KV byte accounting `from_ranks` derives.
    info: WorkerInfo,
    /// Physical runs admitted concurrently, from the ranks' queue depth.
    depth: usize,
    /// The last accepted batch identifier; `submit_batch` refuses a batch
    /// whose identifier does not exceed it.
    last_batch_id: Option<u64>,
    /// Time of construction or of the last submission, joined result, or
    /// group replacement, from which `NEXT_RESULT_DEADLINE` is measured.
    last_progress: Instant,
    /// Admitted batches not yet retired, by batch identifier.
    pending_batches: BTreeMap<u64, PendingBatch>,
    /// Launch arguments. Recovery relaunches the group from them, and
    /// projection and joining read its placement and components.
    process_args: WorkerProcessArgs,
    /// Which component serves each media call across every worker of the
    /// deployment; the executor states it once all workers have reported.
    media_routing: BTreeMap<uniserve_worker_ipc::MediaCall, String>,
    /// Requests that may hold state on the ranks: added by a submitted call
    /// or `Start` command, removed when a batch carrying their `Finish`
    /// command is joined as done. Recovery reports them as lost.
    resident_requests: HashSet<RequestKey>,
    /// Set by recovery and close; `take_readiness_change` reports and clears it.
    readiness_changed: bool,
    /// Set by `close` and by a failed recovery. A closed group accepts no
    /// batch and returns nothing from `poll_batch`.
    closed: bool,
    /// Rank loss is reported after all already-agreed results have been delivered.
    failure: Option<anyhow::Error>,
    /// The deployment's launchers of the hosts this instance does not run
    /// on, shared with the other groups. A launcher terminates the ranks it
    /// started when the registry's last owner closes the connection, so the
    /// group holds it for as long as it holds those ranks.
    launchers: Option<super::launcher::Launchers>,
}

/// A call's identity across request epochs: engine identifier, request
/// identifier, request epoch, and call identifier, in that order.
type CallIdentity = (u64, u64, u64, uniserve_worker_ipc::CallId);

/// A physical batch and its participating ranks share one retirement lifetime.
struct PendingBatch {
    /// The batch as submitted, before rank projection.
    batch: Batch,
    /// Calls not yet joined into a published result.
    remaining: BTreeSet<CallIdentity>,
    /// Every rank that received a projection, by rank.
    ranks: BTreeMap<usize, RankResult>,
}

/// A participating rank's received prefix, independent of transport fragmentation.
struct RankResult {
    /// The calls planned on this rank and whether the rank returned each.
    calls: BTreeMap<CallIdentity, bool>,
    /// Whether the rank returned its result for the batch.
    complete: bool,
    /// The execution error the rank reported for the batch, if any.
    error: Option<WorkerExecError>,
}

/// Returns the request and call identifiers carried by a worker call.
fn call_identity(
    request: uniserve_worker_ipc::RequestKey,
    call_id: uniserve_worker_ipc::CallId,
) -> CallIdentity {
    (
        request.engine_id,
        request.request_id.0,
        request.request_epoch,
        call_id,
    )
}

/// Refuses a placement whose muxer is not on the head's host.
///
/// The artifact a muxer publishes is a POSIX shared-storage object the head
/// opens by name, and such a name resolves in one host's namespace. A muxer
/// placed elsewhere would fail on the first request, so the placement is
/// refused at startup, naming the component and the host it was placed on.
pub(crate) fn refuse_muxer_off_head(
    head: &str,
    ranks: &[crate::WorkerRank],
    components: &BTreeMap<String, crate::executor::ComponentConfig>,
    media_components: &BTreeMap<uniserve_worker_ipc::MediaCall, String>,
) -> anyhow::Result<()> {
    let Some(component) = media_components.get(&uniserve_worker_ipc::MediaCall::Muxing) else {
        return Ok(());
    };
    let placement = components.get(component).with_context(|| {
        format!("component {component} serves muxing but the placement binds no such component")
    })?;
    for &rank in &placement.ranks {
        let node = ranks
            .get(rank)
            .map(|placed| placed.node.as_str())
            .with_context(|| {
                format!("component {component} names rank {rank} outside the placement")
            })?;
        anyhow::ensure!(
            node == head,
            "component {component} serves muxing on rank {rank} on host {node}, but the head is \
             on host {head}, and the artifact it publishes is a shared-storage object named in \
             one host's namespace"
        );
    }
    Ok(())
}

/// Refuses a rank whose loaded checkpoint is not the one the instance serves.
///
/// `reported` holds each rank's checkpoint identity in rank order. When the
/// head derived an expectation from a local checkpoint directory, every rank
/// must have loaded that checkpoint; otherwise every rank must have loaded the
/// checkpoint rank 0 did. The refusal names the rank, the host its placement
/// put it on, and both identities, so the divergent copy can be found.
pub(crate) fn refuse_checkpoint_mismatch(
    expected: Option<&str>,
    ranks: &[crate::WorkerRank],
    reported: &[&str],
) -> anyhow::Result<()> {
    let host = |rank: usize| {
        ranks
            .get(rank)
            .map_or("<unplaced>", |placed| placed.node.as_str())
    };
    let (reference, origin) = match (expected, reported.first()) {
        (Some(identity), _) => (identity, "the head derived checkpoint".to_owned()),
        (None, Some(identity)) => (
            *identity,
            format!("rank 0 on host {} loaded checkpoint", host(0)),
        ),
        (None, None) => return Ok(()),
    };
    for (rank, identity) in reported.iter().enumerate() {
        anyhow::ensure!(
            *identity == reference,
            "physical rank {rank} on host {} loaded checkpoint {identity}, but {origin} {reference}",
            host(rank),
        );
    }
    Ok(())
}

impl WorkerGroup {
    /// Launches every configured rank and exposes the instance after capability agreement.
    pub fn spawn(process_args: WorkerProcessArgs) -> anyhow::Result<Self> {
        Self::spawn_all(vec![process_args])?
            .pop()
            .context("WorkerGroup launch produced no instance")
    }

    /// Launches the configured static rank groups and waits for loaded capabilities.
    ///
    /// Groups are returned in argument order. A failure of any group fails the
    /// whole launch; ranks already spawned are shut down as the values that
    /// own them, and the launcher registry, drop.
    pub fn spawn_all(mut arguments: Vec<WorkerProcessArgs>) -> anyhow::Result<Vec<Self>> {
        // The head derives each group's checkpoint identity once, before any
        // rank starts; every launch and relaunch descriptor then carries it.
        for args in &mut arguments {
            args.derive_checkpoint_identity()?;
        }

        // One launcher per host serves every group of the deployment, so the
        // registry is bound and its hosts awaited once, before any group
        // sends a rank to them.
        let launchers = match arguments.first() {
            Some(first) => super::launcher::LauncherRegistry::for_hosts(
                &first.host,
                arguments.iter().map(|args| args.ranks.as_slice()),
                first.launcher_timeout,
            )?,
            None => None,
        };
        // Every group's processes start before any group waits for reports, so
        // their interpreter and library import overlap.
        let launched = arguments
            .iter()
            .map(|args| args.spawn_ranks(None, launchers.clone()))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let groups = arguments
            .iter()
            .zip(launched)
            .map(|(args, ranks)| args.adopt_ranks(ranks))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut workers = Vec::with_capacity(groups.len());
        for ((mut ranks, launchers), args) in groups.into_iter().zip(arguments) {
            for rank in &mut ranks {
                rank.finish_startup()?;
            }
            workers.push(Self::from_ranks(args, ranks, launchers)?);
        }
        Ok(workers)
    }

    /// Checks that the ranks' startup reports describe one group and builds it.
    ///
    /// Every rank must report valid info naming its launched rank, the group's
    /// world size and worker identity, and a checkpoint
    /// `refuse_checkpoint_mismatch` admits, and must agree with rank 0 apart
    /// from rank-specific fields and numerical settings. The muxer must run on
    /// the head's host, and the ranks' KV regions must cover every layer's KV
    /// heads without a gap. Once all checks pass, no rank's startup may be
    /// cancelled and every rank this process started must still be running.
    fn from_ranks(
        process_args: WorkerProcessArgs,
        mut workers: Vec<RankProcess>,
        launchers: Option<super::launcher::Launchers>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(!workers.is_empty(), "need >= 1 worker");
        let n = workers.len();
        let world_size =
            u32::try_from(n).context("process world size exceeds the IPC representation")?;

        // `info` starts as rank 0's report and becomes the group's; `canonical`
        // is the reference every rank is compared with, without the numerical
        // settings ranks may differ in.
        let mut info = workers[0].info().clone();
        let mut canonical = info.clone();
        canonical.model_dtype.clear();
        canonical.attention_backend.clear();
        canonical.weight_formats.clear();
        canonical.activation_formats.clear();

        // A rank's product storage follows the components it holds, which an
        // asymmetric placement makes rank-specific: a rank that imports a
        // component's product reserves the whole logical allocation for it,
        // and a rank that neither produces nor consumes it reserves nothing.
        // The group reports the smallest, because the engine places work
        // against one pool per group and must not place more than the
        // smallest rank can hold.
        let product_storage = workers
            .iter()
            .map(|worker| worker.info().buffer_pool_bytes)
            .min()
            .unwrap_or(info.buffer_pool_bytes);
        info.buffer_pool_bytes = product_storage;
        canonical.buffer_pool_bytes = product_storage;

        // Checkpoint agreement is checked by name before the generic report
        // comparison, which would otherwise report only that ranks disagree.
        let checkpoints = workers
            .iter()
            .map(|worker| worker.info().checkpoint_identity.as_str())
            .collect::<Vec<_>>();
        refuse_checkpoint_mismatch(
            process_args.checkpoint_identity.as_deref(),
            &process_args.ranks,
            &checkpoints,
        )?;

        for (rank, worker) in workers.iter().enumerate() {
            let rank_info = worker.info();
            rank_info
                .validate()
                .with_context(|| format!("physical rank {rank} reported invalid worker info"))?;
            anyhow::ensure!(
                rank_info.endpoint.rank == rank as u32 && rank_info.world_size == world_size,
                "physical rank {rank} reported topology ({}/{}) for launched topology ({rank}/{world_size})",
                rank_info.endpoint.rank,
                rank_info.world_size,
            );
            let mut normalized = rank_info.clone();
            anyhow::ensure!(
                rank_info.endpoint.worker_id == process_args.worker_id,
                "physical rank {rank} reported another WorkerGroup identity"
            );
            // Rank-specific fields take the reference's values before the
            // comparison: the endpoint, device, transfer backends, product
            // storage, the unit size, and each KV group's layers and head
            // offset.
            normalized.endpoint = canonical.endpoint.clone();
            normalized.device = canonical.device.clone();
            normalized.transfer_backends = canonical.transfer_backends.clone();
            normalized.buffer_pool_bytes = canonical.buffer_pool_bytes;
            if let (Some(local), Some(reference)) = (&mut normalized.kv_cache, &canonical.kv_cache)
            {
                local.unit_bytes = reference.unit_bytes;
                if local.groups.len() == reference.groups.len() {
                    for (group, canonical) in local.groups.iter_mut().zip(&reference.groups) {
                        group.layer_ids.clone_from(&canonical.layer_ids);
                        group.kv_head_offset = canonical.kv_head_offset;
                    }
                }
            }
            // Placement may give ranks different numerical storage; replacement
            // validation (`validate_replacement_info`) compares each rank with
            // its own predecessor instead.
            normalized.model_dtype.clear();
            normalized.attention_backend.clear();
            normalized.weight_formats.clear();
            normalized.activation_formats.clear();
            anyhow::ensure!(
                normalized == canonical,
                "physical rank {rank} worker info disagree with rank 0"
            );
        }

        refuse_muxer_off_head(
            &process_args.host,
            &process_args.ranks,
            &process_args.components,
            &info.media_components,
        )?;

        // Every logical cache layer, numbered from zero without a gap, must
        // belong to one group on every rank that stores it, and in each group
        // every layer must have its KV heads covered from head 0 to
        // `total_kv_heads` by the ranks storing it. Overlapping regions are
        // allowed; a gap is not. The group reports the union of its ranks'
        // layers, which bounds publications, and the largest unit, because
        // the scheduler reserves units shared by every stage.
        if let Some(cache) = &mut info.kv_cache {
            let regions: Vec<_> = workers
                .iter()
                .filter_map(|worker| worker.info().kv_cache.as_ref())
                .collect();
            let mut owners = BTreeMap::new();
            for (index, group) in cache.groups.iter_mut().enumerate() {
                let mut layers: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
                for region in &regions {
                    let local = &region.groups[index];
                    for layer in &local.layer_ids {
                        layers.entry(*layer).or_default().push((
                            local.kv_head_offset,
                            local.kv_head_offset + local.num_kv_heads,
                        ));
                    }
                }
                for (layer, mut heads) in layers {
                    anyhow::ensure!(
                        *owners.entry(layer).or_insert(index) == index,
                        "worker KV layer {layer} belongs to two cache groups"
                    );
                    heads.sort_unstable();
                    let mut covered = 0;
                    for (start, end) in heads {
                        anyhow::ensure!(
                            start <= covered,
                            "worker KV regions leave a logical head gap"
                        );
                        covered = covered.max(end);
                    }
                    anyhow::ensure!(
                        covered == group.total_kv_heads,
                        "worker KV regions do not cover the heads of layer {layer}"
                    );
                }
                group.layer_ids = owners
                    .iter()
                    .filter(|(_, owner)| **owner == index)
                    .map(|(layer, _)| *layer)
                    .collect();
            }
            anyhow::ensure!(
                owners.keys().copied().eq(0..owners.len() as u32),
                "worker KV regions do not cover every logical cache layer"
            );
            cache.unit_bytes = regions
                .iter()
                .map(|region| region.unit_bytes)
                .max()
                .unwrap_or(cache.unit_bytes);
        }

        // Startup is over. A rank's death watcher raises the group's shared
        // startup cancellation; detaching it keeps a later exit from reading
        // as a cancelled startup on the surviving ranks.
        for worker in &mut workers {
            worker.check_worker("WorkerGroup readiness")?;
            worker.set_startup_cancel(None);
        }

        let depth = info.queue_depth.max(1) as usize;
        let buffers = (0..n).map(|_| VecDeque::new()).collect();
        Ok(Self {
            launchers,
            workers,
            buffers,
            depth,
            last_batch_id: None,
            last_progress: Instant::now(),
            pending_batches: BTreeMap::new(),
            media_routing: info.media_components.clone(),
            info,
            process_args,
            resident_requests: HashSet::new(),
            readiness_changed: false,
            closed: false,
            failure: None,
        })
    }

    /// Drains immediately available rank results into per-rank agreement buffers.
    ///
    /// A result that does not match the rank's plan for its batch fails the
    /// drain. A rank error that `record_rank_error` accepts is recorded for
    /// `join_rank_errors`; any other is kept as the group's first `failure`
    /// and ends that rank's drain, so results already buffered can still be
    /// joined before `poll_batch` reports the failure.
    fn pump_once(&mut self) -> anyhow::Result<()> {
        for rank in 0..self.workers.len() {
            loop {
                let result = self.workers[rank].poll_batch(Duration::ZERO);
                match result {
                    Ok(Some(result)) => {
                        let batch_id = result.batch_id;
                        let progress = self
                            .pending_batches
                            .get_mut(&batch_id)
                            .and_then(|batch| batch.ranks.get_mut(&rank))
                            .ok_or_else(|| {
                                anyhow::anyhow!("rank {rank} returned unsubmitted batch {batch_id}")
                            })?;
                        anyhow::ensure!(
                            !progress.complete,
                            "rank {rank} returned after retirement"
                        );
                        let identities = report_call_ids(&result);
                        anyhow::ensure!(
                            result.products.iter().all(|product| identities.contains(
                                &call_identity(
                                    product.product.request_key,
                                    product.product.producer_call_id,
                                )
                            )),
                            "rank {rank} published a product without its call completion"
                        );
                        for identity in identities {
                            let completed = progress.calls.get_mut(&identity).ok_or_else(|| {
                                anyhow::anyhow!(
                                    "rank {rank} returned an unplanned call for batch {batch_id}"
                                )
                            })?;
                            anyhow::ensure!(
                                !*completed,
                                "rank {rank} returned a call more than once for batch {batch_id}"
                            );
                            *completed = true;
                        }
                        // A batch returns one result, so its every call
                        // completes with it.
                        anyhow::ensure!(
                            progress.calls.values().all(|completed| *completed),
                            "rank {rank} ended batch {batch_id} without all completions"
                        );
                        progress.complete = true;
                        self.buffers[rank].push_back(result);
                    }
                    Ok(None) => break,
                    Err(error) => {
                        if let Err(error) = self.record_rank_error(rank, &error) {
                            self.failure.get_or_insert(error);
                            break;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Records a rank's execution error against its pending batch.
    ///
    /// Fails, without recording, when the error is not a `WorkerExecError`,
    /// carries no batch identifier, names a batch or rank with no pending
    /// entry, or differs from an error the rank already reported for that
    /// batch.
    fn record_rank_error(&mut self, rank: usize, error: &anyhow::Error) -> anyhow::Result<()> {
        let execution = error
            .downcast_ref::<WorkerExecError>()
            .ok_or_else(|| anyhow::anyhow!("rank {rank} failed: {error:#}"))?;
        let batch_id = execution.batch_id.ok_or_else(|| {
            anyhow::anyhow!("rank {rank} returned an execution error without a batch identity")
        })?;
        let progress = self
            .pending_batches
            .get_mut(&batch_id)
            .and_then(|batch| batch.ranks.get_mut(&rank))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "rank {rank} returned an execution error for unknown batch {batch_id}"
                )
            })?;
        if let Some(existing) = &progress.error {
            anyhow::ensure!(
                *existing == *execution,
                "rank {rank} returned conflicting errors for batch {batch_id}"
            );
        } else {
            progress.error = Some(execution.clone());
        }
        Ok(())
    }

    /// Replaces the complete rank group after worker loss and invalidates affected requests.
    ///
    /// Returns the [`WorkerFailure`] that reports the lost allocations. Recovery
    /// always invalidates resident work, so the result is an error whether or
    /// not the replacement group started; a failed replacement also closes
    /// this group.
    fn recover_workers(&mut self, cause: &anyhow::Error) -> anyhow::Error {
        // The lost endpoints, resident requests, and reported capabilities
        // are captured before the ranks and bookkeeping are cleared.
        let endpoints = self
            .workers
            .iter()
            .map(|worker| worker.info().endpoint.clone())
            .collect();
        let requests = self.resident_requests.iter().copied().collect();
        let expected = self
            .workers
            .iter()
            .map(|worker| worker.info().clone())
            .collect::<Vec<_>>();
        for worker in &mut self.workers {
            worker.terminate();
        }
        self.workers.clear();
        self.clear_execution();

        let recovery = (|| -> anyhow::Result<()> {
            // The group's ranks on other hosts are stopped by their launchers
            // before the group is relaunched through the same registry, which
            // stays connected for the deployment's other groups.
            if let Some(registry) = &self.launchers {
                super::launcher::lock(registry)?.stop_worker(&self.process_args.worker_id)?;
            }
            let (mut workers, launchers) =
                self.process_args.launch(None, self.launchers.clone())?;
            self.launchers = launchers;
            for worker in &mut workers {
                worker.finish_startup()?;
            }
            anyhow::ensure!(
                workers.len() == expected.len(),
                "replacement rank count changed"
            );
            for (rank, (before, after)) in expected.iter().zip(&workers).enumerate() {
                validate_replacement_info(before, after.info(), rank)?;
            }
            for worker in &mut workers {
                worker.check_worker("WorkerGroup readiness")?;
                worker.set_startup_cancel(None);
            }
            self.install_replacement(workers);
            Ok(())
        })();

        self.readiness_changed = true;
        let recovery_status = match recovery {
            Ok(()) => "rank group recovered".to_owned(),
            Err(error) => {
                self.closed = true;
                format!("rank recovery failed: {error:#}")
            }
        };
        WorkerFailure {
            worker_id: crate::WorkerId(self.process_args.worker_id.clone()),
            endpoints,
            requests,
            retired: Vec::new(),
            buffers: Vec::new(),
            execution: cause.downcast_ref::<WorkerExecError>().cloned(),
            message: format!(
                "WorkerGroup {} lost its resident allocations; {recovery_status}: {cause:#}",
                self.process_args.worker_id
            ),
        }
        .into()
    }

    /// Installs a capability-compatible replacement rank group and resets rank-local state.
    fn install_replacement(&mut self, workers: Vec<RankProcess>) {
        // `validate_replacement_info` has required every replacement rank to
        // match its predecessor apart from the endpoint, so the group's agreed
        // info keeps everything but rank 0's new endpoint.
        self.info.endpoint = workers[0].info().endpoint.clone();
        self.workers = workers;
        self.last_progress = Instant::now();
        let ranks = self.process_args.ranks.len();
        self.buffers = (0..ranks).map(|_| VecDeque::new()).collect();
    }

    /// Clears execution bookkeeping after physical ownership has retired.
    fn clear_execution(&mut self) {
        self.failure = None;
        self.pending_batches.clear();
        self.buffers.iter_mut().for_each(VecDeque::clear);
        self.resident_requests.clear();
    }

    /// Joins mutually agreeing rank reports into one logical physical result.
    ///
    /// Takes the call set `joinable_report_key` selects from every rank
    /// planned for it, validates each report against the batch, and merges
    /// the participants into the output owner's report, which alone carries
    /// public output and forward statistics. The result is `done` once every
    /// call is joined and every rank of the batch has completed; the batch is
    /// then retired. When nothing is joinable, `join_rank_errors` may resolve
    /// a batch whose ranks reported execution errors, returning its error;
    /// otherwise the result is `None`.
    fn try_join(&mut self) -> anyhow::Result<Option<WorkerResult>> {
        let Some((batch_id, output_rank, call_ids)) = self.joinable_report_key() else {
            // Preserve successful partial results already agreed by every rank
            // before retiring the unresolved remainder of a failed batch.
            self.join_rank_errors()?;
            return Ok(None);
        };

        let pending = self
            .pending_batches
            .get_mut(&batch_id)
            .ok_or_else(|| anyhow::anyhow!("joined batch {batch_id} has no pending batch"))?;
        // The ranks planned for every selected call take part; the final join,
        // with an empty call set, takes every rank of the batch.
        let participants = pending
            .ranks
            .iter()
            .filter_map(|(&rank, result)| {
                (call_ids.is_empty() || call_ids.iter().all(|id| result.calls.contains_key(id)))
                    .then_some(rank)
            })
            .collect::<Vec<_>>();
        let batch = &pending.batch;
        let mut out = take_rank_calls(&mut self.buffers[output_rank], batch, &call_ids, true)?;
        validate_and_order_rank_report(batch, &mut out, output_rank)?;
        for rank in participants.into_iter().filter(|rank| *rank != output_rank) {
            let mut report = take_rank_calls(&mut self.buffers[rank], batch, &call_ids, false)?;
            validate_and_order_rank_report(batch, &mut report, rank)?;
            merge_rank_report(batch, &mut out, &report, rank)?;
        }
        let remaining = &mut pending.remaining;
        for identity in &call_ids {
            anyhow::ensure!(
                remaining.remove(identity),
                "joined batch {batch_id} repeated a call"
            );
        }
        let done = remaining.is_empty() && pending.ranks.values().all(|result| result.complete);
        out.done = done;
        if done {
            // Report each rank's accumulated execution time once, after every
            // rank reports. Ranks execute concurrently, so the batch's duration
            // is the maximum of these sums rather than their total.
            out.worker_exec_us = self
                .buffers
                .iter()
                .filter_map(|buffer| {
                    buffer
                        .iter()
                        .filter(|report| report.batch_id == batch_id)
                        .filter_map(|report| report.worker_exec_us)
                        .reduce(u64::saturating_add)
                })
                .max();
            for command in &pending.batch.commands {
                if let BatchCommand::Finish { request_key, .. } = command {
                    self.resident_requests.remove(request_key);
                }
            }
            self.finish_batch(batch_id);
        }
        Ok(Some(out))
    }

    /// Returns the file descriptors that signal worker progress.
    pub(crate) fn progress_fds(&self) -> Vec<libc::pollfd> {
        self.workers
            .iter()
            .flat_map(RankProcess::progress_fds)
            .collect()
    }

    /// Resolves a batch once every rank reports either success or a compatible error.
    ///
    /// Only batches with a rank error whose other ranks have all completed or
    /// failed are resolved, at most one per invocation. When a rank that
    /// succeeded still holds unjoined calls, the ranks disagree: the batch is
    /// retired and a plain error returned. Otherwise every error must be
    /// identical, or a plain error is returned. A fatal error, a
    /// `SchedulerBug` or `InvariantViolation` code, or a batch carrying a
    /// `Free` command returns the `WorkerExecError` itself; `poll_batch`
    /// answers these and the plain errors with recovery. Any other error
    /// retires the batch and returns a `WorkerFailure` naming its unjoined
    /// calls and affected requests, which `poll_batch` returns without
    /// recovery.
    fn join_rank_errors(&mut self) -> anyhow::Result<()> {
        let failing = self
            .pending_batches
            .iter()
            .filter_map(|(&batch_id, pending)| {
                pending
                    .ranks
                    .values()
                    .any(|rank| rank.error.is_some())
                    .then_some(batch_id)
            })
            .collect::<Vec<_>>();
        for batch_id in failing {
            let pending = &self.pending_batches[&batch_id];
            if pending
                .ranks
                .values()
                .any(|rank| rank.error.is_none() && !rank.complete)
            {
                continue;
            }
            let errors = pending
                .ranks
                .iter()
                .filter_map(|(&rank, result)| result.error.as_ref().map(|error| (rank, error)))
                .collect::<Vec<_>>();
            if pending.ranks.values().any(|result| {
                result.error.is_none()
                    && result.calls.keys().any(|id| pending.remaining.contains(id))
            }) {
                let details = errors
                    .iter()
                    .map(|(rank, error)| format!("rank {rank}: {error}"))
                    .collect::<Vec<_>>()
                    .join("; ");
                self.finish_batch(batch_id);
                anyhow::bail!(
                    "worker ranks disagreed between success and failure for batch {batch_id}: {details}"
                );
            }
            let canonical = errors[0].1.clone();
            for (rank, error) in errors.iter().copied().skip(1) {
                anyhow::ensure!(
                    *error == canonical,
                    "rank {rank} returned a different execution error for batch {batch_id}"
                );
            }
            let batch = &self.pending_batches[&batch_id].batch;
            if canonical.fatal
                || matches!(
                    canonical.code.as_deref(),
                    Some("SchedulerBug" | "InvariantViolation")
                )
                || batch
                    .commands
                    .iter()
                    .any(|command| matches!(command, BatchCommand::Free { .. }))
            {
                // A failed physical release cannot return an allocation safely.
                // Replacing this instance proves retirement without retrying
                // the same unsuccessful release indefinitely.
                return Err(canonical.into());
            }
            let pending = &self.pending_batches[&batch_id].remaining;
            let retired = batch
                .calls
                .iter()
                .filter(|call| pending.contains(&call_identity(call.request_key, call.call_id)))
                .map(|call| (batch.batch_id, call.request_key, call.call_id))
                .collect::<Vec<_>>();
            let requests = retired
                .iter()
                .map(|(_, request, _)| *request)
                .chain(batch.commands.iter().map(BatchCommand::request_key))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            let failure = WorkerFailure {
                worker_id: crate::WorkerId(self.process_args.worker_id.clone()),
                endpoints: Vec::new(),
                requests,
                retired,
                buffers: Vec::new(),
                message: canonical.to_string(),
                execution: Some(canonical),
            };
            self.finish_batch(batch_id);
            return Err(failure.into());
        }
        Ok(())
    }

    /// Retires all rank-local tracking for one agreed terminal batch.
    fn finish_batch(&mut self, batch_id: u64) {
        self.pending_batches.remove(&batch_id);
        for buffer in &mut self.buffers {
            buffer.retain(|report| report.batch_id != batch_id);
        }
    }

    /// Join one component's completed work independently of rank and transport framing.
    ///
    /// Batches are considered in identifier order. A batch whose calls are
    /// all joined yields an empty call set once every participating rank has
    /// completed, owned by its lowest participating rank; that final join
    /// takes no call and marks the batch done once ranks without a pending
    /// call, such as ranks that received only commands, have reported.
    /// Otherwise a pending call's output owner is the first rank of its
    /// component, and one of the owner's reports yields every call it carries
    /// that is still pending, belongs to the same component, is planned on
    /// the same ranks, and has been returned by each of them.
    /// Returns the batch identifier, the output owner, and the call set.
    fn joinable_report_key(&self) -> Option<(u64, usize, Vec<CallIdentity>)> {
        for (&batch_id, pending_batch) in &self.pending_batches {
            let pending = &pending_batch.remaining;
            if pending.is_empty() {
                if pending_batch.ranks.values().all(|result| result.complete) {
                    return Some((
                        batch_id,
                        *pending_batch.ranks.first_key_value()?.0,
                        Vec::new(),
                    ));
                }
                continue;
            }
            let batch = &self.pending_batches[&batch_id].batch;
            for call in &batch.calls {
                let identity = call_identity(call.request_key, call.call_id);
                if !pending.contains(&identity) {
                    continue;
                }
                let owner = self.process_args.components[&call.component].ranks[0];
                let members = self.call_members(batch_id, identity);
                for report in self.buffers[owner]
                    .iter()
                    .filter(|report| report.batch_id == batch_id)
                {
                    let shared = report_call_ids(report)
                        .into_iter()
                        .filter(|id| {
                            pending.contains(id)
                                && self.call_members(batch_id, *id) == members
                                && batch.calls.iter().any(|candidate| {
                                    call_identity(candidate.request_key, candidate.call_id) == *id
                                        && candidate.component == call.component
                                })
                                && members.iter().all(|rank| {
                                    self.buffers[*rank].iter().any(|result| {
                                        result.batch_id == batch_id
                                            && result.results.iter().any(|completion| {
                                                call_identity(
                                                    completion.output.request_key,
                                                    completion.output.call_id,
                                                ) == *id
                                            })
                                    })
                                })
                        })
                        .collect::<Vec<_>>();
                    if !shared.is_empty() {
                        return Some((batch_id, owner, shared));
                    }
                }
            }
        }
        None
    }

    /// Returns the ranks, in rank order, whose projection of the batch plans
    /// the call.
    fn call_members(&self, batch_id: u64, identity: CallIdentity) -> Vec<usize> {
        self.pending_batches[&batch_id]
            .ranks
            .iter()
            .filter_map(|(&rank, result)| result.calls.contains_key(&identity).then_some(rank))
            .collect()
    }
}

/// Acknowledgment slots of the ranks reading a media product produced on
/// `rank` of `worker`, where `members` are the ranks executing the call.
///
/// The consuming component may belong to another worker, whose ranks all
/// read the product; on the producing worker the ranks producing their own
/// copy read that copy instead of another rank's.
#[allow(clippy::too_many_arguments)]
pub(crate) fn media_consumer_slots(
    consuming: &[uniserve_worker_ipc::MediaCall],
    routing: &BTreeMap<uniserve_worker_ipc::MediaCall, String>,
    worker: &str,
    components: &BTreeMap<String, crate::ComponentConfig>,
    peers: &BTreeMap<String, BTreeMap<String, crate::ComponentConfig>>,
    transfer: &crate::executor::TransferConfig,
    members: &[usize],
    rank: usize,
    producer: Option<&crate::ComponentConfig>,
) -> Vec<u32> {
    let mut slots = BTreeSet::new();
    for consumer in consuming {
        let Some(name) = routing.get(consumer) else {
            continue;
        };
        let mut owners = peers
            .iter()
            .filter_map(|(id, entries)| entries.get(name).map(|config| (id.as_str(), config)))
            .collect::<Vec<_>>();
        if !owners.iter().any(|(id, _)| *id == worker)
            && let Some(config) = components.get(name)
        {
            owners.push((worker, config));
        }
        for (owner, component) in owners {
            // A round deals its media units to each distributed component's
            // ranks in order, `units_per_rank` each, so a distributed consumer
            // reads only the positions this producer rank wrote. Naming every
            // replica is safe: only the selected consumer claims its slot.
            let dealt = producer
                .filter(|producer| {
                    producer.distribution.is_some() && component.distribution.is_some()
                })
                .and_then(|producer| {
                    producer
                        .ranks
                        .iter()
                        .position(|&member| member == rank)
                        .map(|index| (producer, index))
                });
            let readers: Vec<usize> = match dealt {
                Some((producer, index)) => {
                    let per_producer = producer.units_per_rank.max(1);
                    let per_reader = component.units_per_rank.max(1);
                    (index * per_producer..(index + 1) * per_producer)
                        .filter_map(|position| component.ranks.get(position / per_reader).copied())
                        .collect()
                }
                None => component.ranks.clone(),
            };
            for reader in readers {
                if owner != worker || (reader != rank && !members.contains(&reader)) {
                    slots.insert(transfer.acknowledgment_slot(owner, reader as u32));
                }
            }
        }
    }
    slots.into_iter().collect()
}

impl WorkerGroup {
    /// Acknowledgment slots of the ranks that read the products a call
    /// produces on `rank`, where `members` are the ranks executing the call.
    ///
    /// A video call's readers are the ranks of the components serving the
    /// calls that consume it, less the ranks that produce their own copy: a
    /// consumer reads the copy it holds before any other. Work outside the
    /// video graph is read by the destinations of the rank's transfer edges.
    fn consumer_slots(
        &self,
        call: &uniserve_worker_ipc::Call,
        members: &[usize],
        rank: usize,
    ) -> Vec<u32> {
        let transfer = &self.process_args.transfer;
        let worker = &self.process_args.worker_id;
        let consuming = match call.code {
            uniserve_worker_ipc::CallKind::Media(media_call) => {
                crate::scheduler::consuming_calls(media_call)
            }
            _ => None,
        };
        let Some(consuming) = consuming else {
            return transfer.product_consumers(worker, rank as u32);
        };
        media_consumer_slots(
            consuming,
            &self.media_routing,
            worker,
            &self.process_args.components,
            &self.process_args.peers,
            transfer,
            members,
            rank,
            self.process_args.components.get(&call.component),
        )
    }

    /// The placement this group was launched with: its ranks' hosts and
    /// devices, and its components.
    pub fn placement(
        &self,
    ) -> (
        &[crate::WorkerRank],
        &BTreeMap<String, crate::ComponentConfig>,
    ) {
        (&self.process_args.ranks, &self.process_args.components)
    }

    /// States the deployment-wide media routing once every worker reported.
    pub fn set_media_routing(&mut self, routing: BTreeMap<uniserve_worker_ipc::MediaCall, String>) {
        self.media_routing = routing;
    }
}

/// Restrict a physical invocation to each component's actual members. Request and
/// storage commands retain group-wide visibility, including on otherwise idle
/// ranks; they do not create synthetic computation completions.
fn rank_projection(
    batch: &Batch,
    components: &BTreeMap<String, crate::ComponentConfig>,
    rank_count: usize,
    consumer_slots: impl Fn(&uniserve_worker_ipc::Call, &[usize], usize) -> Vec<u32>,
) -> anyhow::Result<Vec<(usize, Batch)>> {
    let members = batch
        .calls
        .iter()
        .map(|call| {
            let entry = components
                .get(&call.component)
                .with_context(|| format!("unknown component {}", call.component))?;
            // A distributed component deals a call's decoder units to its ranks
            // `units_per_rank` at a time, so only the ranks the call's
            // `max_units` reach take part. `WorkerConfig::validate_members`,
            // which `spawn_ranks` calls, rejects a zero `units_per_rank`.
            let count = if entry.distribution.is_some() {
                let range = batch
                    .decode_ranges
                    .iter()
                    .find(|range| {
                        range.request_key == call.request_key && range.call_id == call.call_id
                    })
                    .context("temporally distributed component requires a decode range")?;
                (range.max_units as usize)
                    .div_ceil(entry.units_per_rank)
                    .min(entry.ranks.len())
            } else {
                entry.ranks.len()
            };
            Ok(&entry.ranks[..count])
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut batches = Vec::new();
    for rank in 0..rank_count {
        let indices = members
            .iter()
            .enumerate()
            .filter_map(|(index, members)| members.contains(&rank).then_some(index))
            .collect::<Vec<_>>();
        if indices.is_empty() && batch.commands.is_empty() {
            continue;
        }
        // A rank receives the batch's projection onto the calls it owns,
        // under the same identity: commands travel to every participating rank.
        // Each call states which ranks read its products, which only the head
        // can derive from the placement.
        let mut projection = batch.clone();
        projection.calls = indices
            .iter()
            .map(|index| {
                let mut call = batch.calls[*index].clone();
                call.consumer_slots = consumer_slots(&call, members[*index], rank);
                call
            })
            .collect();
        // Forward rows are narrowed to the selected calls, and the batch's
        // side tables to what those calls use: their request pool slots,
        // identities, inputs, KV sources, and buffers.
        projection.forward = batch.forward.select(&indices);
        let slots = projection
            .forward
            .request_pool_indices
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        projection
            .block_tables
            .retain(|table| slots.contains(&table.request_pool_idx));
        projection
            .new_cache_units
            .retain(|units| slots.contains(&units.request_pool_idx));
        let identities = projection
            .calls
            .iter()
            .map(|call| (call.request_key, call.call_id))
            .collect::<HashSet<_>>();
        projection
            .latent_params
            .retain(|params| identities.contains(&(params.request_key, params.call_id)));
        projection
            .decode_ranges
            .retain(|range| identities.contains(&(range.request_key, range.call_id)));
        let inputs = projection
            .calls
            .iter()
            .flat_map(|call| call.tensor_inputs().chain(call.predicate.as_ref()))
            .collect::<HashSet<_>>();
        projection
            .input_products
            .retain(|payload| inputs.contains(&payload.product));
        projection.kv_inputs.retain(|publication| {
            projection
                .calls
                .iter()
                .any(|call| call.kv_input == Some(publication.source))
        });
        let buffers = inputs
            .iter()
            .copied()
            .chain(
                projection
                    .calls
                    .iter()
                    .flat_map(|call| call.tensor_outputs()),
            )
            .map(|product| product.buffer_id())
            .collect::<HashSet<_>>();
        projection
            .buffer_allocations
            .retain(|allocation| buffers.contains(&allocation.buffer));
        if projection.calls.is_empty() && projection.commands.is_empty() {
            continue;
        }
        projection.validate()?;
        batches.push((rank, projection));
    }
    Ok(batches)
}

/// Consume selected calls while retaining the rank results that hold the
/// rest. A rank result's aggregate statistics are emitted once, when its final
/// call is consumed; they are never divided or copied.
///
/// An empty `identities` visits every result of the batch without taking a
/// call. `try_join` sets `retain_forward_stats` only when taking from the
/// output owner. Fails when a selected call is missing or a drained result
/// keeps a product, and, with `retain_forward_stats`, when the selection
/// drains more than one result carrying forward statistics.
fn take_rank_calls(
    buffer: &mut VecDeque<WorkerResult>,
    batch: &Batch,
    identities: &[CallIdentity],
    retain_forward_stats: bool,
) -> anyhow::Result<WorkerResult> {
    let mut output = WorkerResult {
        batch_id: batch.batch_id,
        done: false,
        results: Vec::with_capacity(identities.len()),
        products: Vec::new(),
        worker_exec_us: None,
        forward_stats: None,
    };
    for report in buffer
        .iter_mut()
        .filter(|report| report.batch_id == batch.batch_id)
    {
        let selected = report.results.iter().any(|completion| {
            identities.contains(&call_identity(
                completion.output.request_key,
                completion.output.call_id,
            ))
        });
        if !selected && !identities.is_empty() {
            continue;
        }
        output
            .results
            .extend(report.results.extract_if(.., |completion| {
                identities.contains(&call_identity(
                    completion.output.request_key,
                    completion.output.call_id,
                ))
            }));
        output
            .products
            .extend(report.products.extract_if(.., |product| {
                identities.contains(&call_identity(
                    product.product.request_key,
                    product.product.producer_call_id,
                ))
            }));
        if report.results.is_empty() {
            anyhow::ensure!(
                report.products.is_empty(),
                "rank result retained an unowned product"
            );
            if retain_forward_stats && report.forward_stats.is_some() {
                anyhow::ensure!(
                    output.forward_stats.is_none(),
                    "canonical call selection spans statistics fragments"
                );
                output.forward_stats = report.forward_stats.take();
            }
        }
    }
    // A drained result stays buffered while it carries `worker_exec_us`, which
    // `try_join` sums per rank once the batch is done; `finish_batch` then
    // removes it.
    buffer.retain(|report| {
        report.batch_id != batch.batch_id
            || !report.results.is_empty()
            || report.worker_exec_us.is_some()
    });
    anyhow::ensure!(
        output.results.len() == identities.len(),
        "rank result omitted selected calls"
    );
    Ok(output)
}

/// Returns the call identifiers carried by a rank report.
fn report_call_ids(report: &WorkerResult) -> Vec<CallIdentity> {
    let mut ids = report
        .results
        .iter()
        .map(|output| call_identity(output.output.request_key, output.output.call_id))
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

/// Join cooperative completion reports into the designated output owner's report.
/// Every participant agrees on semantic completion; only the output owner
/// publishes host products.
fn merge_rank_report(
    batch: &Batch,
    canonical_report: &mut WorkerResult,
    participant_report: &WorkerResult,
    rank: usize,
) -> anyhow::Result<()> {
    let batch_id = batch.batch_id;
    if participant_report.batch_id != batch_id {
        anyhow::bail!(
            "rank {rank} completion report batch ID mismatch while joining pending batch {batch_id}: got {}",
            participant_report.batch_id
        );
    }
    if participant_report.results.len() != canonical_report.results.len() {
        anyhow::bail!(
            "rank {rank} report for batch {batch_id} has {} completions, expected {}",
            participant_report.results.len(),
            canonical_report.results.len()
        );
    }
    for (completion_index, (canonical, actual)) in canonical_report
        .results
        .iter_mut()
        .zip(&participant_report.results)
        .enumerate()
    {
        anyhow::ensure!(
            batch
                .calls
                .iter()
                .any(|call| call.request_key == canonical.output.request_key
                    && call.call_id == canonical.output.call_id),
            "rank join received an unplanned call for batch {batch_id}"
        );
        if let Err(error) = merge_completion_record(canonical, actual) {
            anyhow::bail!(
                "rank {rank} report for batch {batch_id} completion {completion_index} differs from the output owner: {error:#}"
            );
        }
    }
    for product in &participant_report.products {
        anyhow::ensure!(
            batch
                .calls
                .iter()
                .flat_map(|call| call.tensor_outputs())
                .any(|output| output == &product.product),
            "rank {rank} published locations for an undeclared product"
        );
        if let Some(existing) = canonical_report
            .products
            .iter_mut()
            .find(|existing| existing.product == product.product)
        {
            existing.merge_locations(product)?;
        } else {
            canonical_report.products.push(product.clone());
        }
    }
    Ok(())
}

/// Validates a rank report and orders completions to match the submitted call sequence.
fn validate_and_order_rank_report(
    batch: &Batch,
    report: &mut WorkerResult,
    rank: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        report.batch_id == batch.batch_id,
        "rank {rank} returned pending batch {} for pending batch {}",
        report.batch_id,
        batch.batch_id
    );
    let report_count = report.results.len();
    let returned = report_call_ids(report).into_iter().collect::<BTreeSet<_>>();
    let mut ordered = Vec::with_capacity(report_count);
    for planned in &batch.calls {
        let identity = call_identity(planned.request_key, planned.call_id);
        if !returned.contains(&identity) {
            continue;
        }
        let index = report
            .results
            .iter()
            .position(|completion| {
                completion.output.request_key == planned.request_key
                    && completion.output.call_id == planned.call_id
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "rank {rank} omitted a call for pending batch {} collective {}",
                    batch.batch_id,
                    batch.collective_seq
                )
            })?;
        let completion = report.results.swap_remove(index);
        anyhow::ensure!(
            completion.output.status != uniserve_worker_ipc::CallStatus::Predicated
                || report.products.iter().all(|publication| {
                    publication.product.request_key != completion.output.request_key
                        || publication.product.producer_call_id != completion.output.call_id
                }),
            "predicated call published a tensor"
        );
        if planned.code
            == uniserve_worker_ipc::CallKind::Transfer(uniserve_worker_ipc::TransferMode::KvPublish)
            && completion.output.status == uniserve_worker_ipc::CallStatus::Ok
        {
            let publication = completion
                .output
                .kv_output
                .as_ref()
                .context("successful KV publication has no transfer descriptor")?;
            anyhow::ensure!(
                Some(publication.source) == planned.kv_output,
                "KV publication differs from its declared output"
            );
            let bytes = publication.tensors().try_fold(0_u64, |sum, tensor| {
                Ok::<_, uniserve_worker_ipc::ValidationError>(
                    sum.saturating_add(tensor.validate()?),
                )
            })?;
            anyhow::ensure!(
                bytes <= planned.bounds.max_transfer_bytes,
                "KV publication exceeds its transfer-byte bound"
            );
        }
        ordered.push(completion);
    }
    anyhow::ensure!(
        ordered.len() == report_count,
        "rank {rank} returned an unplanned call for pending batch {}",
        batch.batch_id
    );
    report.results = ordered;
    Ok(())
}

/// Merge one participant into the output owner's result. Accepted progress and
/// allocation generations agree; KV publications contribute immutable rank locations.
fn merge_completion_record(
    canonical: &mut CallResult,
    rank_completion: &CallResult,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        canonical.output.request_key == rank_completion.output.request_key
            && canonical.output.call_id == rank_completion.output.call_id,
        "completion identity diverged"
    );
    anyhow::ensure!(
        canonical.output.committed_tokens.as_slice()
            == rank_completion.output.committed_tokens.as_slice()
            && canonical.output.status == rank_completion.output.status
            && canonical.output.position == rank_completion.output.position
            && canonical.output.kv_visible_len == rank_completion.output.kv_visible_len
            && canonical.output.kv_computed_len == rank_completion.output.kv_computed_len
            && canonical.output.num_completed_steps == rank_completion.output.num_completed_steps
            && canonical.output.finish_flags == rank_completion.output.finish_flags,
        "completion result fields diverged"
    );
    anyhow::ensure!(
        rank_completion.media.as_ref().is_ok_and(Option::is_none)
            && rank_completion.output.sampled_logprob.is_none()
            && rank_completion.output.top_logprobs.is_empty()
            && rank_completion.output.prompt_logprobs.is_empty()
            && rank_completion.output.candidate_logprobs.is_empty(),
        "non-designated rank returned public output"
    );
    anyhow::ensure!(
        canonical.output.product_generations == rank_completion.output.product_generations,
        "designated-rank product generations diverged"
    );
    match (
        &mut canonical.output.kv_output,
        &rank_completion.output.kv_output,
    ) {
        (Some(stored), Some(publication)) => stored.merge_locations(publication)?,
        (None, None) => {}
        _ => anyhow::bail!("rank KV publication presence diverged"),
    }
    Ok(())
}

/// Validates that a replacement rank preserves the established worker contract.
fn validate_replacement_info(
    expected: &WorkerInfo,
    actual: &WorkerInfo,
    rank: usize,
) -> anyhow::Result<()> {
    actual
        .validate()
        .with_context(|| format!("replacement rank {rank} reported invalid worker info"))?;
    let mut normalized_expected = expected.clone();
    anyhow::ensure!(
        actual.endpoint.worker_id == expected.endpoint.worker_id
            && actual.endpoint.rank == rank as u32
            && actual.endpoint.incarnation != expected.endpoint.incarnation,
        "replacement rank {rank} did not establish a new endpoint incarnation"
    );
    normalized_expected.endpoint = actual.endpoint.clone();
    anyhow::ensure!(
        normalized_expected == *actual,
        "replacement rank {rank} worker info changed"
    );
    Ok(())
}

impl WorkerGroup {
    /// Reports whether this instance is open and holds its rank group.
    ///
    /// `close` and a failed recovery make it permanently unready.
    pub fn is_ready(&self) -> bool {
        !self.closed && !self.workers.is_empty()
    }

    /// Physical submission slots not yet owned by accepted batches.
    pub(crate) fn available_slots(&self) -> usize {
        if self.is_ready() {
            self.depth.saturating_sub(self.pending_batches.len())
        } else {
            0
        }
    }

    /// Reports and clears whether recovery or close ran since the last call.
    pub(crate) fn take_readiness_change(&mut self) -> bool {
        std::mem::take(&mut self.readiness_changed)
    }

    /// Returns each rank's host and whether its device exports a fabric handle.
    ///
    /// A transfer edge crosses hosts when its endpoints report different hosts,
    /// and it can only do so if both devices export a handle the other host can
    /// import.
    pub fn rank_fabric_reach(&self) -> Vec<(String, bool)> {
        self.workers
            .iter()
            .map(|rank| {
                let info = rank.info();
                (info.endpoint.node.clone(), info.fabric_handles)
            })
            .collect()
    }

    /// Returns the group's agreed worker metadata.
    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Return the same bindings used to initialize this instance's transports.
    pub(crate) fn transfer_config(&self) -> &crate::executor::TransferConfig {
        &self.process_args.transfer
    }

    /// Exposes accepted calls not yet joined into a published result.
    pub(crate) fn inflight_calls(&self) -> impl Iterator<Item = &uniserve_worker_ipc::Call> {
        self.pending_batches.values().flat_map(|pending| {
            pending.batch.calls.iter().filter(|call| {
                pending
                    .remaining
                    .contains(&call_identity(call.request_key, call.call_id))
            })
        })
    }

    /// Returns the loaded physical endpoints used for transfer binding.
    pub(crate) fn rank_info(&self, rank: usize) -> Option<&WorkerInfo> {
        self.workers.get(rank).map(RankProcess::info)
    }

    /// Submit each call to its component members and lifetime commands to the rank group.
    ///
    /// Returns `WouldBlock` with the batch when the group holds no ranks or
    /// all `depth` slots are held. A closed group, a batch identifier that does
    /// not exceed the previous one, an invalid batch, or a failed projection
    /// or input binding returns `Failed` with no state change; a failed input
    /// binding carries the first refusing rank's `UnroutableInputs`. A rank
    /// submission that fails after the batch is recorded replaces the rank
    /// group and returns `Failed` with the recovery's `WorkerFailure`.
    pub fn submit_batch(&mut self, batch: Batch) -> Result<(), BatchSubmitError> {
        if self.closed {
            return Err(BatchSubmitError::Failed(anyhow::anyhow!(
                "WorkerGroup is closed"
            )));
        }
        if self
            .last_batch_id
            .is_some_and(|last| batch.batch_id <= last)
        {
            return Err(BatchSubmitError::Failed(anyhow::anyhow!(
                "batch IDs must increase in submission order"
            )));
        }
        if !self.is_ready() || self.pending_batches.len() >= self.depth {
            return Err(BatchSubmitError::WouldBlock(Box::new(batch)));
        }
        batch
            .validate()
            .map_err(anyhow::Error::from)
            .map_err(BatchSubmitError::Failed)?;
        let rank_batches = rank_projection(
            &batch,
            &self.process_args.components,
            self.workers.len(),
            |call, members, rank| self.consumer_slots(call, members, rank),
        )
        .and_then(|batches| {
            batches
                .into_iter()
                .map(|(rank, mut batch)| {
                    self.process_args.transfer.bind_inputs(
                        &mut batch.input_products,
                        &mut batch.kv_inputs,
                        &self.workers[rank].info().endpoint,
                    )?;
                    Ok((rank, batch))
                })
                .collect::<anyhow::Result<Vec<_>>>()
        })
        .map_err(BatchSubmitError::Failed)?;

        // The batch's requests are recorded as resident before any rank
        // receives it, so the recovery that follows a failed rank submission
        // reports them as lost too.
        let batch_id = batch.batch_id;
        let requests = batch
            .calls()
            .map(|call| call.request_key)
            .collect::<Vec<_>>();
        self.last_batch_id = Some(batch_id);
        self.resident_requests.extend(requests);
        self.resident_requests
            .extend(batch.commands.iter().filter_map(|command| match command {
                BatchCommand::Start { request } => Some(request.request_key),
                _ => None,
            }));
        let ranks = rank_batches
            .iter()
            .map(|(rank, batch)| {
                (
                    *rank,
                    RankResult {
                        calls: batch
                            .calls
                            .iter()
                            .map(|call| (call_identity(call.request_key, call.call_id), false))
                            .collect(),
                        complete: false,
                        error: None,
                    },
                )
            })
            .collect();
        self.pending_batches.insert(
            batch_id,
            PendingBatch {
                remaining: batch
                    .calls
                    .iter()
                    .map(|call| call_identity(call.request_key, call.call_id))
                    .collect(),
                batch,
                ranks,
            },
        );
        self.last_progress = Instant::now();

        for (rank, batch) in rank_batches {
            if let Err(error) = self.workers[rank].submit_batch(batch) {
                // The group admitted the batch against the queue depth every
                // rank was launched with, so a rank that hands it back, for a
                // full queue or a full socket send queue, fails the group
                // rather than applying backpressure.
                let error = match error {
                    BatchSubmitError::WouldBlock(_) => {
                        anyhow::anyhow!(
                            "physical rank {rank} rejected an admitted cooperative batch as full"
                        )
                    }
                    BatchSubmitError::Failed(error) => {
                        error.context(format!("submit to rank {rank} failed"))
                    }
                };
                return Err(BatchSubmitError::Failed(self.recover_workers(&error)));
            }
        }
        Ok(())
    }

    /// Waits for rank progress and returns the next fully agreed physical result.
    ///
    /// Returns `None` when `timeout` passes without a joinable result, and at
    /// once for a closed group. The `WorkerFailure` of an agreed execution
    /// error that `join_rank_errors` retires with its batch is returned as is
    /// and the group stays usable. Any other error, including rank loss, a
    /// protocol violation, an agreed error `join_rank_errors` does not retire,
    /// and an expired `NEXT_RESULT_DEADLINE`, replaces the rank group and
    /// returns the recovery's `WorkerFailure`.
    pub fn poll_batch(&mut self, timeout: Duration) -> anyhow::Result<Option<WorkerResult>> {
        if self.closed {
            return Ok(None);
        }
        match self.poll_progress(timeout) {
            Ok(report) => Ok(report),
            Err(error) if error.is::<WorkerFailure>() => Err(error),
            Err(error) => Err(self.recover_workers(&error)),
        }
    }

    fn poll_progress(&mut self, timeout: Duration) -> anyhow::Result<Option<WorkerResult>> {
        let deadline = Instant::now() + timeout;
        loop {
            // A drain can observe both final replies and rank death. Join every
            // complete reply before recovery invalidates outstanding ownership.
            if let Some(result) = self.try_join()? {
                self.last_progress = Instant::now();
                return Ok(Some(result));
            }
            if let Some(error) = self.failure.take() {
                return Err(error);
            }
            self.pump_once()?;
            if let Some(result) = self.try_join()? {
                self.last_progress = Instant::now();
                return Ok(Some(result));
            }
            if let Some(error) = self.failure.take() {
                return Err(error);
            }
            self.check_progress_deadline()?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            crate::worker::park_descriptors(&self.progress_fds(), remaining)?;
        }
    }

    fn check_progress_deadline(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.pending_batches.is_empty() || self.last_progress.elapsed() < NEXT_RESULT_DEADLINE,
            "cooperative workers produced no progress within {:?}",
            NEXT_RESULT_DEADLINE
        );
        Ok(())
    }

    fn release_closed_resources(&mut self) {
        // A closed group never triggers recovery: `poll_batch` returns early
        // and `submit_batch` refuses.
        self.closed = true;
        self.workers.clear();
        self.clear_execution();
        self.readiness_changed = true;
    }

    /// Closes every rank in the physical worker group.
    pub fn close(&mut self) -> anyhow::Result<()> {
        self.closed = true;
        let mut first_error = None;
        for worker in self.workers.iter_mut() {
            if let Err(error) = worker.close()
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        self.release_closed_resources();
        if let Some(error) = first_error {
            Err(error)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{refuse_checkpoint_mismatch, refuse_muxer_off_head};
    use crate::WorkerRank;
    use crate::executor::ComponentConfig;
    use std::collections::BTreeMap;
    use uniserve_worker_ipc::MediaCall;

    fn placement(nodes: &[&str]) -> Vec<WorkerRank> {
        nodes
            .iter()
            .enumerate()
            .map(|(index, node)| WorkerRank {
                node: (*node).to_owned(),
                device: format!("cuda:{index}"),
            })
            .collect()
    }

    fn components(muxer_ranks: Vec<usize>) -> BTreeMap<String, ComponentConfig> {
        BTreeMap::from([(
            "muxer".to_owned(),
            ComponentConfig::parallel(muxer_ranks, Default::default()),
        )])
    }

    fn muxing(component: &str) -> BTreeMap<MediaCall, String> {
        BTreeMap::from([(MediaCall::Muxing, component.to_owned())])
    }

    #[test]
    fn a_muxer_on_the_head_host_is_admitted() {
        let ranks = placement(&["rank-0", "rank-0", "rank-1", "rank-1"]);
        refuse_muxer_off_head("rank-0", &ranks, &components(vec![0]), &muxing("muxer"))
            .expect("a muxer on the head's host serves the artifact the head opens");
        // A deployment without a muxer publishes no artifact and is unaffected.
        refuse_muxer_off_head("rank-0", &ranks, &components(vec![3]), &BTreeMap::new())
            .expect("a placement without muxing has no artifact to place");
    }

    #[test]
    fn a_muxer_off_the_head_host_is_refused_by_name() {
        let ranks = placement(&["rank-0", "rank-0", "rank-1", "rank-1"]);
        let error = refuse_muxer_off_head("rank-0", &ranks, &components(vec![3]), &muxing("muxer"))
            .expect_err("a muxer on another host cannot publish an artifact the head opens");
        let message = format!("{error:#}");
        assert!(
            message.contains("muxer"),
            "the refusal names the component: {message}"
        );
        assert!(
            message.contains("rank-1"),
            "the refusal names the muxer's host: {message}"
        );
        assert!(
            message.contains("rank-0"),
            "the refusal names the head's host: {message}"
        );

        let error =
            refuse_muxer_off_head("rank-0", &ranks, &components(vec![0]), &muxing("output"))
                .expect_err("a muxing component the placement does not bind is refused");
        assert!(format!("{error:#}").contains("output"));
    }

    const SERVED: &str = "0000000000000000000000000000000000000000000000000000000000000000";
    const OTHER: &str = "1111111111111111111111111111111111111111111111111111111111111111";

    #[test]
    fn ranks_that_loaded_the_same_checkpoint_are_admitted() {
        let ranks = placement(&["rank-0", "rank-1"]);
        refuse_checkpoint_mismatch(Some(SERVED), &ranks, &[SERVED, SERVED])
            .expect("every rank loaded the checkpoint the head derived");
        refuse_checkpoint_mismatch(None, &ranks, &[SERVED, SERVED])
            .expect("without an expectation, ranks agreeing with rank 0 are admitted");
        // A stub launch has no checkpoint on either side.
        refuse_checkpoint_mismatch(None, &ranks, &["", ""])
            .expect("a launch without a checkpoint has nothing to compare");
    }

    #[test]
    fn a_checkpoint_the_head_did_not_derive_is_refused_by_name() {
        let ranks = placement(&["rank-0", "rank-1"]);
        let error = refuse_checkpoint_mismatch(Some(SERVED), &ranks, &[SERVED, OTHER])
            .expect_err("a rank that loaded another checkpoint is refused");
        let message = format!("{error:#}");
        assert!(
            message.contains("rank 1"),
            "the refusal names the rank: {message}"
        );
        assert!(
            message.contains("rank-1"),
            "the refusal names the rank's host: {message}"
        );
        assert!(
            message.contains(OTHER),
            "the refusal names what was loaded: {message}"
        );
        assert!(
            message.contains(SERVED),
            "the refusal names the expectation: {message}"
        );

        // Rank 0 itself may be the divergent copy.
        let error = refuse_checkpoint_mismatch(Some(SERVED), &ranks, &[OTHER, SERVED])
            .expect_err("rank 0 is held to the head's expectation too");
        assert!(format!("{error:#}").contains("rank 0 on host rank-0"));
    }

    #[test]
    fn ranks_disagreeing_without_an_expectation_are_refused_against_rank_0() {
        let ranks = placement(&["rank-0", "rank-1"]);
        let error = refuse_checkpoint_mismatch(None, &ranks, &[SERVED, OTHER])
            .expect_err("ranks must load one checkpoint even when the head derived none");
        let message = format!("{error:#}");
        assert!(message.contains("rank 1 on host rank-1"), "{message}");
        assert!(message.contains("rank 0 on host rank-0"), "{message}");
        assert!(
            message.contains(OTHER) && message.contains(SERVED),
            "{message}"
        );
    }
}
