//! One rank's process and channel, from spawn to shutdown.
//!
//! `PendingRank::spawn_rank` writes the rank's launch descriptor and starts
//! the Python worker here, or hands the launch to the host's launcher
//! (`launcher::RemoteHost`) when the placement puts the rank on another host.
//! Once the rank reports its endpoint through `registration::RankRegistry`,
//! `PendingRank::adopt` binds a `RankChannel` to it and yields a
//! `RankProcess`, which `WorkerGroup` (in `instance`) drives:
//! `finish_startup` validates the rank's startup report, `submit_batch` and
//! `poll_batch` exchange batches and results, and `close_ranks` or
//! `terminate` retires the rank.
//!
//! The host exchanges FlatBuffers descriptors and bounded result values while
//! tensors, KV pages, and latent storage remain worker-resident.
//!
//! The module also defines the launch settings `WorkerProcessArgs` carries
//! (`LaneConfig`, `FlashInferBackend`), that type's defaults, and the launch
//! descriptor built from it.

use crate::executor::WorkerResult;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use super::BatchSubmitError;
use crate::executor::WorkerExecError;
use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uniserve_worker_ipc::{Batch, DEFAULT_COMPONENT, WorkerInfo, WorkerRequest, WorkerResponse};
use uniserve_worker_ipc::{Frame, Outstanding, RankChannel};

/// How long the head waits for a rank's socket channel to accept.
///
/// A rank binds its address before it registers, so this covers accepting an
/// already-bound connection rather than waiting for the rank to start. A
/// shared-storage connect does not use it.
const CHANNEL_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

use crate::worker::WorkerProcessArgs;
use crate::worker::death_watch::DeathWatcher;

/// Deadline for `finish_startup` to send the startup info request, retrying
/// until the rank's end of the channel is connected.
///
/// It bounds only the send. The rank answers once its model is built and
/// warmed up, and that wait has no deadline here (see
/// `RankProcess::wait_pending_response`).
const WORKER_CONNECT_TIMEOUT: Duration = Duration::from_secs(300);
/// Deadline for sending the shutdown request in `request_close`.
///
/// The channel was connected at startup, so a missing server connection at
/// this point means the rank has dropped off, and shutdown fails fast rather
/// than waiting for the startup deadline.
const WORKER_SEND_TIMEOUT: Duration = Duration::from_secs(30);
/// Maximum time `request_close` spends draining in-flight responses from a
/// still-alive worker before it sends the shutdown request. A hung (alive but
/// unresponsive) worker must not be able to block shutdown forever.
const WORKER_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);
/// How long the ranks closed together by `close_ranks` have, from the last
/// shutdown request, to acknowledge it, finish their teardown and exit. A
/// local process still running afterwards is killed, which abandons whatever
/// its teardown had not yet released.
const WORKER_EXIT_GRACE: Duration = Duration::from_secs(30);
/// Longest wait between checks for the exit of a rank `close_ranks` is
/// closing. A local process exit raises no wake once the rank's death watcher
/// has stopped, and a launcher reports a remote exit within its own
/// 100-millisecond read timeout.
const EXIT_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// Interval between progress logs while a rank prepares during startup.
const STARTUP_LOG_INTERVAL: Duration = Duration::from_secs(30);
/// Longest wait between liveness checks (`RankProcess::check_worker`) while
/// waiting on a rank, which bounds how late a local rank's exit is noticed
/// without a death watcher.
const WORKER_CHECK_INTERVAL: Duration = Duration::from_millis(500);

pub use uniserve_worker_ipc::LaneConfig;

/// FlashInfer implementation selected in [`WorkerProcessArgs`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlashInferBackend {
    /// Lets the worker select a compatible backend.
    #[default]
    Auto,
    /// Uses FlashAttention 2 kernels.
    Fa2,
    /// Uses FlashAttention 3 kernels.
    Fa3,
}

impl FlashInferBackend {
    /// Returns the stable command-line spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Fa2 => "fa2",
            Self::Fa3 => "fa3",
        }
    }
}

impl std::str::FromStr for FlashInferBackend {
    type Err = FlashInferBackendParseError;

    /// Parses the spelling [`FlashInferBackend::as_str`] produces.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "fa2" => Ok(Self::Fa2),
            "fa3" => Ok(Self::Fa3),
            _ => Err(FlashInferBackendParseError(value.to_owned())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported FlashInfer backend {0:?}")]
/// Error returned for an unsupported FlashInfer backend name.
pub struct FlashInferBackendParseError(String);

impl Default for WorkerProcessArgs {
    /// Returns worker launch settings suitable for a single local rank.
    fn default() -> Self {
        Self {
            worker_id: "worker".into(),
            python: "python3".into(),
            launcher_timeout: std::time::Duration::from_secs(120),
            model: String::new(),
            base_model: None,
            ranks: vec![crate::WorkerRank {
                node: "localhost".into(),
                device: "cuda:0".into(),
            }],
            host: "localhost".into(),
            // One local rank running one component. A launch replaces this
            // with the components the model being served declared.
            components: crate::WorkerConfig::single_component(DEFAULT_COMPONENT, 1),
            role: crate::WorkerRole::Model,
            peers: Default::default(),
            stub: false,
            queue_depth: 2,
            req_slot_cap: 1 << 20,
            resp_slot_cap: 8 << 20,
            kv_token_capacity: None,
            block_size: None,
            max_batch_calls: 128,
            max_batch_tokens: 16_384,
            // The engine's default running-request limit plus the largest
            // row reserve a runtime keeps outside it.
            max_request_pool_size: (crate::scheduler::DEFAULT_MAX_NUM_SEQS
                + crate::scheduler::MAX_FLOW_PREFIX_ROWS) as u32,
            attention_backend: uniserve_worker_ipc::AttentionBackend::Auto,
            capability_groups: Vec::new(),
            transfer: Default::default(),
            load_format: "auto".to_string(),
            download_dir: None,
            load_threads: None,
            checksum_manifest: None,
            model_dtype: uniserve_core::ModelDtype::BFloat16,
            quantization_config: serde_json::json!({}),
            kv_cache_dtype: None,
            kv_storage_fraction: 0.70,
            mesh: None,
            distributed_backend: None,
            lanes: Vec::new(),
            graph_policy: "auto".into(),
            decode_graph_batch_sizes: None,
            prefill_cuda_graph: true,
            prefill_outputs: true,
            prefill_graph_token_sizes: None,
            flow_cuda_graph: true,
            flow_graph_batch_sizes: None,
            flow_graph_shapes: None,
            video_text_capacities: None,
            canvas_sampling: None,
            flashinfer_workspace_size: 512 * 1024 * 1024,
            flashinfer_use_tensor_core: None,
            flashinfer_decode_backend: FlashInferBackend::Fa2,
            flashinfer_prefill_backend: FlashInferBackend::Auto,
            flashinfer_decode_split_tile_size: None,
            flashinfer_prefill_split_tile_size: None,
            flashinfer_disable_split_kv: false,
            max_model_len: 8192,
            max_video_seconds: 15.0,
            max_condition_rows: 0,
            ffmpeg: "ffmpeg".into(),
            min_video_seconds: None,
            expert_parallel: None,
            expert_microbatches: 1,
        }
    }
}

impl WorkerProcessArgs {
    /// Returns one rank's index among the ranks on its own host and that
    /// host's rank count.
    ///
    /// These are the host-relative `LOCAL_RANK` and `LOCAL_WORLD_SIZE`
    /// values: a rank's position among the ranks the placement puts on the
    /// same node, and how many ranks that node holds. They coincide with the
    /// global values while an instance occupies one host. `rank` must index
    /// `self.ranks`.
    fn host_slot(&self, rank: u32) -> anyhow::Result<(u32, u32)> {
        let rank = rank as usize;
        let node = &self.ranks[rank].node;
        let resident = |placed: &&crate::WorkerRank| &placed.node == node;

        // The rank's host index counts the ranks placed before it on its node.
        let index = self.ranks[..rank].iter().filter(resident).count();
        let count = self.ranks.iter().filter(resident).count();
        Ok((
            u32::try_from(index).context("a host's rank index exceeds u32")?,
            u32::try_from(count).context("a host's rank count exceeds u32")?,
        ))
    }

    /// Returns the number of ranks in this group.
    fn world_size(&self) -> u32 {
        self.ranks.len() as u32
    }

    /// Names the channel mechanism `rank` serves its endpoint over.
    ///
    /// A rank on the head's host can offer shared storage; a rank placed
    /// elsewhere has none to offer and serves a socket.
    fn channel_transport(&self, rank: u32) -> &'static str {
        if self.ranks[rank as usize].node == self.host {
            uniserve_worker_ipc::SHARED_STORAGE_CHANNEL
        } else {
            uniserve_worker_ipc::SOCKET_CHANNEL
        }
    }

    /// Builds the typed launch descriptor the worker process consumes.
    ///
    /// Every tuning value is stated explicitly, so the launching side is the
    /// single source of defaults and the worker never re-derives one. The keys
    /// become attributes of the namespace `WorkerProcessArgs.from_namespace`
    /// reads, and the worker refuses a descriptor missing any key listed in
    /// `REQUIRED_FIELDS` (`uniserve_worker.bootstrap.cli`). Only the process
    /// identity and the descriptor's path travel on argv.
    ///
    /// `rank` must index `self.ranks`. Fails when a host-relative rank value
    /// does not fit `u32` or the component configuration does not serialize.
    fn launch_descriptor(
        &self,
        rank: u32,
        registration: &str,
        rendezvous: Option<&str>,
        rendezvous_listen_fd: Option<std::os::fd::RawFd>,
        expert_listen_fd: Option<std::os::fd::RawFd>,
    ) -> anyhow::Result<serde_json::Value> {
        let depth = self.queue_depth.max(1);
        let max_payload = self.req_slot_cap.max(self.resp_slot_cap).max(1);
        // Resolve mechanism ownership from the physical rank's incident edges.
        let (transfer_backends, export_backends) =
            self.transfer.rank_backends(&self.worker_id, rank);
        let names = |backends: &std::collections::BTreeSet<crate::executor::TransferBackend>| {
            backends
                .iter()
                .map(|backend| backend.as_str())
                .collect::<Vec<_>>()
                .join(",")
        };

        let mut fields = serde_json::Map::new();
        fields.insert("worker_id".into(), json!(self.worker_id));
        fields.insert("registration_address".into(), json!(registration));
        // The placement decides the mechanism and the rank names the endpoint:
        // a rank on the head's host can offer shared storage, a rank elsewhere
        // cannot, and only the head knows where a rank was placed.
        fields.insert(
            "channel_transport".into(),
            json!(self.channel_transport(rank)),
        );
        // A rank cannot name the ranks that read what it publishes: it knows
        // its own component, not which component consumes its products, and a
        // product is consumed in a later batch than the one producing it. The
        // head states a product's readers on the call that produces it and
        // gives the rank its own slot here. A consumer writes its own slot's
        // word in every chunk or segment it reads; the slots a producer
        // watches travel on each producing call.
        fields.insert(
            "acknowledgment_slot".into(),
            json!(self.transfer.acknowledgment_slot(&self.worker_id, rank)),
        );
        // Which of those slots are on this rank's host decides the mechanism
        // a host product is published over: shared storage reaches the host,
        // the rank channel reaches the rest.
        fields.insert(
            "host_slots".into(),
            json!(self.transfer.host_slots(&self.worker_id, rank)),
        );
        // Readiness is a producer synchronize only where an interprocess event
        // cannot carry it, which is exactly where a consumer is on another
        // host. Only the placement knows that.
        fields.insert(
            "products_cross_hosts".into(),
            json!(self.transfer.products_cross_hosts(&self.worker_id, rank)),
        );
        fields.insert("queue_depth".into(), json!(depth));
        fields.insert("ipc_payload_cap".into(), json!(max_payload));
        fields.insert("model".into(), json!(self.model));
        if let Some(base) = &self.base_model {
            fields.insert("base_model".into(), json!(base));
        }
        fields.insert("device".into(), json!(self.ranks[rank as usize].device));
        fields.insert("rank".into(), json!(rank));
        fields.insert("local_rank".into(), json!(self.host_slot(rank)?.0));
        fields.insert("world_size".into(), json!(self.world_size()));
        fields.insert("components".into(), serde_json::to_value(&self.components)?);
        // Every component any group of the deployment places, so each rank
        // resolves the capabilities the deployment as a whole selects, such
        // as which of a checkpoint's denoisers it serves.
        let deployment = self
            .peers
            .values()
            .flat_map(|components| components.keys())
            .chain(self.components.keys())
            .collect::<std::collections::BTreeSet<_>>();
        fields.insert("deployment_components".into(), json!(deployment));
        // Null lets the worker serve every capability group it implements.
        fields.insert(
            "supported_calls".into(),
            if self.capability_groups.is_empty() {
                Value::Null
            } else {
                json!(self.capability_groups.join(","))
            },
        );
        fields.insert("transfer_backends".into(), json!(names(&transfer_backends)));
        fields.insert("export_backends".into(), json!(names(&export_backends)));
        // Every rank of a group connects to the collective store at this
        // address. The group's first rank serves it on the socket it inherits
        // at the named descriptor rather than binding the port itself; a
        // first rank placed elsewhere has the number filled in by the
        // launcher that holds its host's reservation.
        fields.insert("rendezvous_address".into(), json!(rendezvous));
        fields.insert(
            uniserve_core::launch::RENDEZVOUS_LISTEN_FD.into(),
            json!(rendezvous_listen_fd),
        );
        // A replica sharing its experts joins the expert-parallel world at the
        // store world rank 0 serves on the socket it inherits at the named
        // descriptor; the other replicas receive only the address.
        fields.insert(
            "expert_parallel".into(),
            match &self.expert_parallel {
                Some(placement) => json!({
                    "rank": placement.rank + rank,
                    "size": placement.size,
                    "attention_ranks": placement.attention_ranks,
                    "address": placement.address,
                    "listen_fd": expert_listen_fd,
                    "exchange": placement.exchange,
                }),
                None => Value::Null,
            },
        );
        fields.insert("role".into(), json!(self.role));
        fields.insert(
            "expert_microbatches".into(),
            json!(self.expert_microbatches),
        );
        fields.insert(
            "distributed_backend".into(),
            json!(self.distributed_backend),
        );
        fields.insert("mesh".into(), json!(self.mesh));
        fields.insert("lane".into(), json!(self.lanes));
        // The worker refuses `no_model` without `allow_stub`, so a stub
        // launch states both.
        fields.insert("no_model".into(), json!(self.stub));
        fields.insert("allow_stub".into(), json!(self.stub));
        fields.insert("load_format".into(), json!(self.load_format));
        fields.insert("download_dir".into(), json!(self.download_dir));
        fields.insert("load_threads".into(), json!(self.load_threads));
        fields.insert("checksum_manifest".into(), json!(self.checksum_manifest));
        fields.insert("model_dtype".into(), json!(self.model_dtype.as_str()));
        fields.insert(
            "quantization_config".into(),
            self.quantization_config.clone(),
        );
        fields.insert(
            "kv_cache_dtype".into(),
            json!(self.kv_cache_dtype.as_ref().map(|value| value.as_str())),
        );
        fields.insert("kv_memory_fraction".into(), json!(self.kv_storage_fraction));
        fields.insert("kv_token_capacity".into(), json!(self.kv_token_capacity));
        fields.insert(
            "attention_backend".into(),
            json!(self.attention_backend.as_name()),
        );
        fields.insert("block_size".into(), json!(self.block_size));
        fields.insert("max_batch_calls".into(), json!(self.max_batch_calls));
        fields.insert("max_batch_tokens".into(), json!(self.max_batch_tokens));
        fields.insert(
            "max_request_pool_size".into(),
            json!(self.max_request_pool_size),
        );
        fields.insert("max_model_len".into(), json!(self.max_model_len));
        fields.insert("max_video_seconds".into(), json!(self.max_video_seconds));
        fields.insert("max_condition_rows".into(), json!(self.max_condition_rows));
        fields.insert("ffmpeg".into(), json!(self.ffmpeg));
        fields.insert("min_video_seconds".into(), json!(self.min_video_seconds));
        fields.insert("graph_policy".into(), json!(self.graph_policy));
        fields.insert(
            "decode_graph_batch_sizes".into(),
            json!(self.decode_graph_batch_sizes),
        );
        fields.insert("prefill_cuda_graph".into(), json!(self.prefill_cuda_graph));
        fields.insert("prefill_outputs".into(), json!(self.prefill_outputs));
        fields.insert(
            "prefill_graph_token_sizes".into(),
            json!(self.prefill_graph_token_sizes),
        );
        fields.insert("flow_cuda_graph".into(), json!(self.flow_cuda_graph));
        fields.insert(
            "flow_graph_batch_sizes".into(),
            json!(self.flow_graph_batch_sizes),
        );
        fields.insert("flow_graph_shapes".into(), json!(self.flow_graph_shapes));
        fields.insert(
            "video_text_capacities".into(),
            json!(self.video_text_capacities),
        );
        fields.insert(
            "canvas_sampling".into(),
            serde_json::to_value(self.canvas_sampling)?,
        );
        fields.insert(
            "flashinfer_workspace_size".into(),
            json!(self.flashinfer_workspace_size),
        );
        fields.insert(
            "flashinfer_use_tensor_core".into(),
            json!(self.flashinfer_use_tensor_core),
        );
        fields.insert(
            "flashinfer_decode_backend".into(),
            json!(self.flashinfer_decode_backend.as_str()),
        );
        fields.insert(
            "flashinfer_prefill_backend".into(),
            json!(self.flashinfer_prefill_backend.as_str()),
        );
        fields.insert(
            "flashinfer_decode_split_tile_size".into(),
            json!(self.flashinfer_decode_split_tile_size),
        );
        fields.insert(
            "flashinfer_prefill_split_tile_size".into(),
            json!(self.flashinfer_prefill_split_tile_size),
        );
        fields.insert(
            "flashinfer_disable_split_kv".into(),
            json!(self.flashinfer_disable_split_kv),
        );
        Ok(Value::Object(fields))
    }
}

/// One adopted rank: its channel, its process when this engine started it,
/// and the requests outstanding on it.
///
/// Dropping it closes the rank alone through `close_ranks`.
pub(super) struct RankProcess {
    client: RankChannel,
    /// The rank's validated startup report; the default value until
    /// `finish_startup` stores it.
    info: WorkerInfo,
    /// The startup cancellation flag shared by the ranks launched together,
    /// which any of their death watchers sets. `WorkerGroup` detaches it with
    /// `set_startup_cancel(None)` once the group is ready.
    startup_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// The rank's process, when this engine started it. A rank placed on
    /// another host is started by that host's launcher and its process is
    /// never visible here; its exit surfaces through the channel once the
    /// connection closes, as `death` in its wake events or as a transport
    /// error from a receive.
    child: Option<Child>,
    /// Keeps the launch descriptor's directory alive for the rank's
    /// lifetime; dropping it deletes the file.
    _launch_descriptor: tempfile::TempDir,
    /// Most requests outstanding at once. `adopt` also sizes the channel
    /// with it (see `RankChannel::connect`), and `finish_startup` requires
    /// the worker to report the same `queue_depth`.
    depth: usize,
    rank: u32,
    world_size: u32,
    expected_components: std::collections::BTreeMap<String, uniserve_core::ComponentConfig>,
    /// Submitted batches awaiting a response, by message identity.
    pending: HashMap<u64, PendingRecord>,
    /// Routed results and errors, in arrival order, that `poll_batch` has
    /// not returned yet.
    ready: VecDeque<anyhow::Result<WorkerResult>>,
    /// Next message identity to assign. Identities start at 1 and are never
    /// zero; `route` treats a zero header identity as absent.
    next_message_id: u64,
    /// How far the rank's retirement has progressed.
    retirement: Retirement,
    /// Edge-triggered worker-death watcher: fires the channel's death wake
    /// and sets the startup cancellation flag when the child exits. `None`
    /// for a rank started on another host, off Linux, when
    /// `DeathWatcher::spawn` failed, and once `request_close` or `terminate`
    /// has stopped it; a local rank's exit is then noticed by `check_worker`.
    death_watcher: Option<DeathWatcher>,
}

/// How far a rank's retirement has progressed.
enum Retirement {
    /// Neither `request_close` nor `terminate` has run.
    Serving,
    /// `request_close` has asked the rank to shut down, and `await_close`
    /// has not yet waited for it. `acknowledgment` is the shutdown request
    /// while its answer is outstanding; it is `None` when the rank had
    /// already exited or the request could not be sent.
    Closing { acknowledgment: Option<Outstanding> },
    /// The rank was closed or terminated; nothing remains to do.
    Retired,
}

/// One launched rank between its start and its endpoint report.
///
/// The engine cannot bind the rank's channel until the rank names its own
/// endpoint, so everything the channel needs is retained here meanwhile.
pub(super) struct PendingRank {
    /// Released by adoption; a local rank still held here is killed when
    /// this value drops, as it does when the launch fails.
    child: PendingChild,
    rank: u32,
    world_size: u32,
    /// Channel sizing `adopt` passes to `RankChannel::connect`.
    depth: usize,
    /// Largest frame payload in bytes, in either direction.
    max_payload: usize,
    /// Moves by adoption alongside the process it cancels.
    startup_abort: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Retains the launch descriptor until the rank has read it; adoption
    /// transfers it to the `RankProcess`, which keeps it for the rank's
    /// lifetime.
    launch_descriptor: tempfile::TempDir,
    components: std::collections::BTreeMap<String, crate::executor::ComponentConfig>,
}

/// The process of a launched rank that has not been adopted yet, when this
/// engine started it.
struct PendingChild(Option<Child>);

impl PendingChild {
    /// Hands the process to the adopting rank, which owns its termination.
    fn release(mut self) -> Option<Child> {
        self.0.take()
    }
}

/// One submitted request awaiting its response.
struct PendingRecord {
    kind: OutstandingKind,
    pending: Outstanding,
}

/// Drops an answered request's channel handle and returns what it was sent for.
///
/// On a shared-storage channel, dropping the handle frees one of the
/// iceoryx2 client's active-request slots, of which it holds at most the
/// rank's queue depth.
fn release_consumed_request(record: PendingRecord) -> OutstandingKind {
    let PendingRecord { kind, pending } = record;
    drop(pending);
    kind
}

/// What an outstanding request was sent for; only batch submissions are
/// tracked in `RankProcess::pending`.
enum OutstandingKind {
    Batch { batch_id: u64 },
}

/// Caching-allocator variables PyTorch reads, the first taking precedence.
const ALLOCATOR_VARIABLES: [&str; 2] = ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"];

/// The transparent-huge-page option of mimalloc, which PyTorch builds in as
/// its CPU allocator on aarch64 Linux.
const HOST_ALLOCATOR_THP_VARIABLE: &str = "MIMALLOC_ALLOW_THP";

/// Returns the allocator variables every rank is launched with, given a
/// lookup into the head's environment.
///
/// Every rank serves varying shapes from expandable caching-allocator
/// segments, unless the head's environment already configures the caching
/// allocator under either name, in which case each rank receives exactly
/// the head's variables. Tensor export never depends on the caching
/// allocator: a device product is exported from the rank's own VMM arena or
/// copied into its bounded VMM pool, both reserved outside the allocator.
///
/// Every rank also keeps its host heap out of transparent huge pages unless
/// the head sets `MIMALLOC_ALLOW_THP` itself. mimalloc otherwise advises its
/// 1 GiB arenas for huge pages. On a kernel with 64 KiB base pages a huge
/// page is 512 MiB, and each time khugepaged collapses one it holds the
/// rank's memory map for about 90 ms, which stalls the rank's service
/// thread. With the option off, mimalloc disables transparent huge pages
/// for the rank process.
///
/// The variables are set explicitly on each rank's command, even where a
/// local child would inherit them, because a launcher on another host
/// receives only the variables the command sets and applies them over its
/// own environment.
fn allocator_environment(
    head: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Vec<(&'static str, std::ffi::OsString)> {
    let mut configured: Vec<_> = ALLOCATOR_VARIABLES
        .into_iter()
        .filter_map(|name| head(name).map(|value| (name, value)))
        .collect();
    if configured.is_empty() {
        configured.push(("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True".into()));
    }
    configured.push((
        HOST_ALLOCATOR_THP_VARIABLE,
        head(HOST_ALLOCATOR_THP_VARIABLE).unwrap_or_else(|| "0".into()),
    ));
    configured
}

/// Bound listening sockets a rank inherits and serves collective stores on.
#[derive(Default)]
pub(crate) struct RankSockets {
    /// The group's rendezvous store, served by the group's first rank.
    pub(crate) group: Option<std::net::TcpListener>,
    /// The expert-parallel world's store, served by that world's rank 0.
    pub(crate) experts: Option<std::net::TcpListener>,
}

impl PendingRank {
    /// Spawns one rank, which reports its endpoint to `registration`.
    ///
    /// The rank's channel is bound afterwards from that report, so nothing here
    /// names the endpoint and nothing waits for model readiness. `rendezvous`
    /// is the group's collective store address, and `sockets.group` the
    /// socket this rank serves that store on when it is the group's first rank
    /// and this process spawns it; the rank inherits the socket and this
    /// process keeps no copy. With `remote`, the launch goes to that host's
    /// launcher and no process is started here. `startup_abort` is the
    /// startup cancellation flag shared by the ranks launched together.
    /// `sockets.experts` is the socket this rank serves its expert-parallel
    /// world's store on, when it is that world's rank 0.
    ///
    /// Fails when the descriptor directory cannot be created, the rank's
    /// host-relative values or its descriptor cannot be built, the descriptor
    /// cannot be written, the remote delivery fails, or the process cannot be
    /// spawned.
    pub(crate) fn spawn_rank(
        args: &WorkerProcessArgs,
        rank: u32,
        rendezvous: Option<&str>,
        sockets: RankSockets,
        remote: Option<&mut super::launcher::RemoteHost<'_>>,
        startup_abort: std::sync::Arc<std::sync::atomic::AtomicBool>,
        registration: &str,
    ) -> anyhow::Result<Self> {
        let depth = args.queue_depth.max(1);
        let max_payload = args.req_slot_cap.max(args.resp_slot_cap).max(1);
        let world_size = args.world_size();

        let descriptor_directory = tempfile::Builder::new()
            .prefix("uniserve-worker-launch")
            .tempdir()
            .context("creating the worker launch descriptor directory")?;
        let descriptor_path = descriptor_directory.path().join("launch.json");

        let mut cmd = Command::new(&args.python);
        cmd.arg("-m")
            .arg("uniserve_worker.main")
            .arg("--worker-id")
            .arg(&args.worker_id)
            .arg("--rank")
            .arg(rank.to_string())
            .arg("--world-size")
            .arg(world_size.to_string())
            .arg("--launch-descriptor")
            .arg(&descriptor_path);

        // Torch and the numerical libraries read the host-relative pair; the
        // global pair names the rank's place in the whole process world.
        let (local_rank, local_world_size) = args.host_slot(rank)?;
        cmd.env("RANK", rank.to_string())
            .env("WORLD_SIZE", world_size.to_string())
            .env("LOCAL_RANK", local_rank.to_string())
            .env("LOCAL_WORLD_SIZE", local_world_size.to_string());

        // The allocator settings are stated even for a local child, which
        // would inherit them, so that a remote launch carries them as well.
        cmd.envs(allocator_environment(|name| std::env::var_os(name)));

        // The engine's working directory leads the worker's import path.
        if let Ok(cwd) = std::env::current_dir() {
            let pp = std::env::var("PYTHONPATH").unwrap_or_default();
            cmd.env("PYTHONPATH", format!("{}:{}", cwd.display(), pp));
        }

        // The command owns the store socket from here and closes this
        // process's copy when it is dropped, after the spawn below.
        let rendezvous_listen_fd = sockets
            .group
            .map(|listener| uniserve_core::launch::inherit_listener(&mut cmd, listener));
        let expert_listen_fd = sockets
            .experts
            .map(|listener| uniserve_core::launch::inherit_listener(&mut cmd, listener));

        // One typed descriptor carries every launch value. argv keeps the
        // process identity and the descriptor's location so a running worker
        // remains identifiable from the process table.
        let descriptor = args.launch_descriptor(
            rank,
            registration,
            rendezvous,
            rendezvous_listen_fd,
            expert_listen_fd,
        )?;
        std::fs::write(&descriptor_path, serde_json::to_vec_pretty(&descriptor)?)
            .context("writing the worker launch descriptor")?;

        // A rank placed on another host is started by that host's launcher,
        // which needs the same descriptor and environment this process would
        // have used. The head derives both once, here, for either path.
        if let Some(remote) = remote {
            remote.deliver(
                rank,
                world_size,
                &args.python,
                &args.worker_id,
                &descriptor,
                &cmd,
            )?;
            return Ok(Self {
                child: PendingChild(None),
                rank,
                world_size,
                depth,
                max_payload,
                startup_abort,
                launch_descriptor: descriptor_directory,
                components: args.components.clone(),
            });
        }
        let child = cmd.spawn().context("spawning python worker")?;

        Ok(Self {
            child: PendingChild(Some(child)),
            rank,
            world_size,
            depth,
            max_payload,
            startup_abort,
            launch_descriptor: descriptor_directory,
            components: args.components.clone(),
        })
    }

    /// Returns the rank this process was launched as.
    pub(crate) fn rank(&self) -> u32 {
        self.rank
    }

    /// Fails by name when the rank exited before reporting its endpoint.
    ///
    /// `RankRegistry::collect` calls this, through the `alive` callback
    /// `adopt_ranks` passes it, while no report is waiting. Reaps the process
    /// when it has exited.
    pub(crate) fn check_alive(&mut self) -> anyhow::Result<()> {
        // A rank started by another host's launcher has no process here. Its
        // exit before registration reaches the head as a report from the
        // launcher that owns it, which `adopt_ranks` checks separately.
        let Some(child) = self.child.0.as_mut() else {
            return Ok(());
        };
        match child.try_wait() {
            Ok(None) => Ok(()),
            Ok(Some(status)) => Err(anyhow::anyhow!(
                "rank {} exited with {status} before reporting its endpoint",
                self.rank
            )),
            Err(error) => Err(error).context("checking a launched rank"),
        }
    }

    /// Binds this rank's channel to the endpoint and mechanism the rank reported.
    ///
    /// The returned `RankProcess` owns the rank's process, when this engine
    /// spawned it, and a death watcher for it when one can be started. Its
    /// `info` stays the default until `RankProcess::finish_startup` runs.
    pub(crate) fn adopt(self, transport: &str, endpoint: &str) -> anyhow::Result<RankProcess> {
        let rank = self.rank;
        // The process stays owned here until the channel exists, so a failed
        // connection terminates the rank instead of orphaning it.
        let client = RankChannel::connect(
            transport,
            endpoint,
            self.max_payload,
            self.depth,
            CHANNEL_CONNECT_TIMEOUT,
        )
        .with_context(|| format!("connecting to the channel rank {rank} reported"))?;
        // The mechanism and endpoint are the rank's choice, so the head records
        // what it bound.
        tracing::info!(
            rank,
            transport,
            endpoint,
            "bound rank channel from its registration"
        );

        let PendingRank {
            child,
            rank,
            world_size,
            depth,
            max_payload: _,
            startup_abort,
            launch_descriptor,
            components,
        } = self;
        let child = child.release();
        // A watcher observes a process identifier, so only a rank this engine
        // started has one. A rank elsewhere falls to the bounded liveness
        // probe, which reads its channel rather than its process.
        let death_watcher = child.as_ref().and_then(|child| {
            DeathWatcher::spawn(
                child.id(),
                client.death_wake(),
                std::sync::Arc::clone(&startup_abort),
            )
        });
        Ok(RankProcess {
            client,
            info: WorkerInfo::default(),
            startup_cancel: Some(startup_abort),
            child,
            _launch_descriptor: launch_descriptor,
            depth,
            rank,
            world_size,
            expected_components: components,
            pending: HashMap::new(),
            ready: VecDeque::new(),
            next_message_id: 1,
            retirement: Retirement::Serving,
            death_watcher,
        })
    }
}

impl Drop for PendingChild {
    /// A rank that never reported an endpoint has no channel to close through,
    /// so a failed launch terminates its process here rather than leaking it.
    fn drop(&mut self) {
        if let Some(child) = self.0.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl RankProcess {
    /// Waits for the rank's startup report and checks it against the launch.
    ///
    /// The rank answers the info request only once its model is built and
    /// warmed up, so once the request is sent this blocks for the rank's
    /// whole preparation, bounded by the rank's liveness and the startup
    /// cancellation flag rather than a deadline. The report must be valid,
    /// match the launched queue depth, rank, world size, and components, and
    /// state the resolved model dtype and attention backend. On success it
    /// becomes `info`; on failure `info` is unchanged.
    pub(crate) fn finish_startup(&mut self) -> anyhow::Result<()> {
        let message_id = self.alloc_call_id();
        let mut request = WorkerRequest::info();
        request.set_call_id(Some(message_id));
        let pending =
            self.send_request_with_timeout(&request, "Worker startup", WORKER_CONNECT_TIMEOUT)?;
        let response = self
            .wait_pending_response(&pending, "Worker startup")?
            .decode_response()?;

        let info = match response {
            WorkerResponse::Info { info, .. } => info,
            WorkerResponse::Error { error, .. } => {
                bail!("Worker startup failed: {}", error.message)
            }
            WorkerResponse::Result { result, .. } => {
                // Receiving claims any media the result published, so the
                // shared-memory objects are released rather than left named.
                drop(WorkerResult::receive(result));
                bail!("unexpected startup result response");
            }
            other => bail!("unexpected startup response: {:?}", other.kind()),
        };

        info.validate()
            .context("worker reported invalid worker info during startup")?;
        let host_depth = self.depth as u32;
        anyhow::ensure!(
            info.queue_depth == host_depth,
            "worker queue_depth {} does not match launched depth {}",
            info.queue_depth,
            host_depth
        );
        anyhow::ensure!(
            info.endpoint.rank == self.rank && info.world_size == self.world_size,
            "worker process rank/world_size ({}/{}) does not match launched topology ({}/{})",
            info.endpoint.rank,
            info.world_size,
            self.rank,
            self.world_size
        );
        anyhow::ensure!(
            info.components
                .iter()
                .map(|entry| (entry.name.clone(), entry.config.clone()))
                .collect::<std::collections::BTreeMap<_, _>>()
                == self.expected_components,
            "worker resolved component configuration disagrees with configuration"
        );
        anyhow::ensure!(
            !info.model_dtype.is_empty() && !info.attention_backend.is_empty(),
            "worker omitted resolved numerical settings"
        );

        self.check_worker("Worker startup")?;
        self.info = info;
        tracing::info!(?self.info, "worker ready");
        Ok(())
    }

    /// Allocates the next worker message identifier.
    fn alloc_call_id(&mut self) -> u64 {
        let id = self.next_message_id;
        self.next_message_id += 1;
        id
    }

    /// Replaces the startup cancellation flag `check_worker` consults.
    ///
    /// `WorkerGroup` passes `None` once the group is ready, so a later exit
    /// no longer reads as a cancelled startup on the surviving ranks.
    pub(crate) fn set_startup_cancel(
        &mut self,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) {
        self.startup_cancel = cancel;
    }

    /// Aborts peers when an unfinished deployment is being dismantled.
    /// Ready ranks have detached the flag, so ordinary shutdown is unaffected.
    pub(crate) fn cancel_startup(&self) {
        if let Some(cancel) = &self.startup_cancel {
            cancel.store(true, std::sync::atomic::Ordering::Release);
        }
    }

    /// Terminates a failed or cancelled rank and waits for process-owned resources to retire.
    ///
    /// Sends no shutdown request and discards outstanding and routed results.
    pub(crate) fn terminate(&mut self) {
        self.retirement = Retirement::Retired;
        // The watcher stops before the kill, so the intended exit raises no
        // death wake and no startup cancellation.
        self.death_watcher.take();
        // A rank this engine started is killed and reaped here. A rank
        // elsewhere has no process here: `WorkerGroup` recovery stops it
        // through `LauncherRegistry::stop_worker`, and a launcher terminates
        // every rank it started when its connection to the head closes.
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.pending.clear();
        self.ready.clear();
    }

    /// Checks child liveness and cancellation of an unfinished startup.
    ///
    /// Fails when the startup cancellation flag is set, when the local
    /// process has exited (reaping it), or when querying the process fails. A
    /// rank on another host is checked for cancellation only.
    pub(super) fn check_worker(&mut self, context: &str) -> anyhow::Result<()> {
        if self
            .startup_cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(std::sync::atomic::Ordering::Acquire))
        {
            bail!("worker startup cancelled during {context}");
        }
        if let Some(child) = self.child.as_mut()
            && let Some(status) = child.try_wait()?
        {
            bail!("worker process exited during {context}: {status}");
        }
        Ok(())
    }

    /// Sends a request, waiting at most `WORKER_SEND_TIMEOUT` for a
    /// connected server.
    fn send_request_checked(
        &mut self,
        req: &WorkerRequest,
        context: &str,
    ) -> anyhow::Result<Outstanding> {
        self.send_request_with_timeout(req, context, WORKER_SEND_TIMEOUT)
    }

    /// Waits for an IPC server connection and submits a request before the deadline.
    ///
    /// Retries while a socket send queue is full or a shared-storage request
    /// reached no connected server, checking liveness between attempts. Wakes
    /// consumed by the wait between attempts are not reported to the caller.
    fn send_request_with_timeout(
        &mut self,
        req: &WorkerRequest,
        context: &str,
        timeout: Duration,
    ) -> anyhow::Result<Outstanding> {
        let deadline = Instant::now() + timeout;
        loop {
            let pending = match self.client.send_request_attempt(req) {
                Ok(pending) => Some(pending),
                Err(uniserve_worker_ipc::IpcError::WouldBlock) => None,
                Err(error) => return Err(error.into()),
            };
            // An unconnected shared-storage request reaches no one; it drops at
            // the end of this iteration, which frees its request slot.
            if let Some(pending) = pending
                && self.client.is_connected(&pending)
            {
                return Ok(pending);
            }
            self.check_worker(context)?;
            if Instant::now() >= deadline {
                bail!("worker IPC service had no connected server during {context}");
            }
            self.client.wait_wake(Duration::from_millis(20))?;
        }
    }

    /// Waits for one startup response while checking child liveness and reporting progress.
    fn wait_pending_response(
        &mut self,
        pending: &Outstanding,
        context: &str,
    ) -> anyhow::Result<Frame> {
        let started = Instant::now();
        let mut last_log = started;
        let mut last_worker_check = started;
        // Loading checkpoints and preparing numerical shapes have no
        // model-independent completion deadline, so this wait has none; any
        // launch deadline belongs to the caller of the launch. Rank death and
        // cancellation remain observable throughout preparation.
        loop {
            if let Some(frame) = self.client.try_recv_response(pending)? {
                return Ok(frame);
            }
            if last_worker_check.elapsed() >= WORKER_CHECK_INTERVAL {
                self.check_worker(context)?;
                last_worker_check = Instant::now();
            }
            if last_log.elapsed() >= STARTUP_LOG_INTERVAL {
                tracing::info!(
                    elapsed_secs = started.elapsed().as_secs(),
                    "worker still busy during {context}"
                );
                last_log = Instant::now();
            }
            let until_check = WORKER_CHECK_INTERVAL.saturating_sub(last_worker_check.elapsed());
            let until_log = STARTUP_LOG_INTERVAL.saturating_sub(last_log.elapsed());
            self.client.wait_wake(until_check.min(until_log))?;
        }
    }

    /// Drains ready IPC responses and routes them into `ready`.
    ///
    /// Returns how many responses were routed and the wakes drained before
    /// the scan. A transport, execution, or routing error stops the scan and
    /// is returned; responses routed before it stay in `ready`.
    fn drain_ready(&mut self) -> anyhow::Result<(usize, uniserve_worker_ipc::WakeEvents)> {
        let wakes = self.client.drain_wakes()?;
        let ids = self.pending.keys().copied().collect::<Vec<_>>();
        let mut drained = 0usize;
        for message_id in ids {
            let Some(record) = self.pending.get(&message_id) else {
                continue;
            };
            let Some(frame) = self.client.try_recv_response(&record.pending)? else {
                continue;
            };
            let record = self.pending.remove(&message_id).ok_or_else(|| {
                anyhow::anyhow!("pending record {message_id} disappeared while routing response")
            })?;
            // Consuming the response ends this IPC request. On a
            // shared-storage channel, release its iceoryx2 active-request slot
            // before routing the result.
            let kind = release_consumed_request(record);
            self.route(message_id, kind, frame)?;
            drained += 1;
        }
        Ok((drained, wakes))
    }

    /// Decodes one batch response and appends its result to `ready`.
    ///
    /// The result is received, which claims any media it published
    /// (`WorkerResult::receive`), before the correlation checks run, so a
    /// rejected response still releases that storage. Returns an error and
    /// appends nothing when the frame does not decode, either stated message
    /// identity differs from `message_id`, the worker answered with an
    /// execution error (as a `WorkerExecError`) or an unexpected response
    /// kind, a product locator names an endpoint other than this rank's
    /// reported one, or the result's batch identity differs from the
    /// submitted batch.
    fn route(
        &mut self,
        message_id: u64,
        kind: OutstandingKind,
        frame: Frame,
    ) -> anyhow::Result<()> {
        let OutstandingKind::Batch { batch_id } = kind;
        let response = frame.decode_response()?;
        let echoed = response.message_id();
        let report = match response {
            WorkerResponse::Result { result, .. } => Ok(WorkerResult::receive(result)),
            WorkerResponse::Error { error, .. } => Err(WorkerExecError {
                batch_id: Some(batch_id),
                fatal: error.fatal,
                code: error.code,
                message: error.message,
                phase: error.phase,
                route: error.route,
                calls: error.calls,
            }
            .into()),
            other => Err(anyhow::anyhow!(
                "unexpected execute response kind: {:?}",
                other.kind()
            )),
        };
        // A zero header identity means the response carries none
        // (`header_for_response`); an identity that is stated must match.
        if frame.header.message_id != 0 && frame.header.message_id != message_id {
            bail!(
                "worker response call id mismatch: expected {message_id}, got {}",
                frame.header.message_id
            );
        }
        if let Some(echoed) = echoed
            && echoed != message_id
        {
            bail!("worker response echoed call id {echoed}, expected {message_id}");
        }
        let report = report?;

        // A product must name this rank's reported endpoint, which identifies
        // the incarnation whose startup report `finish_startup` accepted.
        for product in &report.products {
            anyhow::ensure!(
                product
                    .value
                    .locators()
                    .all(|locator| locator.source == self.info.endpoint),
                "worker published a product from an unbound rank incarnation"
            );
        }
        anyhow::ensure!(
            report.batch_id == batch_id,
            "worker result batch id mismatch: expected {batch_id}, got {}",
            report.batch_id
        );
        self.ready.push_back(Ok(report));
        Ok(())
    }

    /// Returns the descriptors that signal this rank's progress.
    ///
    /// A shared-storage channel multiplexes every wake onto one listener; a
    /// socket channel carries only results and raises the engine's own wakes
    /// on a second descriptor.
    pub(crate) fn progress_fds(&self) -> Vec<libc::pollfd> {
        self.client.progress_fds()
    }
}

impl RankProcess {
    /// Returns metadata for the physical worker.
    pub(super) fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Submits one physical run and records its outstanding message identity.
    ///
    /// Hands the batch back as `WouldBlock` when `depth` requests are already
    /// outstanding or a socket send queue is full; the batch is not recorded
    /// as pending then.
    /// Fails when draining earlier responses or the send fails.
    pub(super) fn submit_batch(&mut self, batch: Batch) -> Result<(), BatchSubmitError> {
        // Answered requests free their slots before the depth check.
        self.drain_ready().map_err(BatchSubmitError::Failed)?;
        // Without this check, a shared-storage send beyond the client's
        // request limit would fail as a transport error rather than
        // `WouldBlock`.
        if self.pending.len() >= self.depth {
            return Err(BatchSubmitError::WouldBlock(Box::new(batch)));
        }
        let batch_id = batch.batch_id;
        let message_id = self.alloc_call_id();
        let mut req = WorkerRequest::submit(batch);
        req.set_call_id(Some(message_id));
        let pending = match self.client.send_request_attempt(&req) {
            Ok(pending) => pending,
            Err(uniserve_worker_ipc::IpcError::WouldBlock) => {
                return Err(BatchSubmitError::WouldBlock(match req {
                    WorkerRequest::Submit { batch, .. } => batch,
                    _ => unreachable!(),
                }));
            }
            Err(error) => return Err(BatchSubmitError::Failed(error.into())),
        };
        self.pending.insert(
            message_id,
            PendingRecord {
                kind: OutstandingKind::Batch { batch_id },
                pending,
            },
        );
        Ok(())
    }

    /// Returns completed responses before failing outstanding work on disconnect.
    ///
    /// Returns the next entry of `ready`, waiting up to `timeout`, and
    /// `Ok(None)` when the timeout passes with nothing ready. An entry is a
    /// result or the error that draining or routing a response produced.
    /// Results already received are returned before a rank exit, startup
    /// cancellation, a channel death wake, or a failed wait fails the call.
    pub(super) fn poll_batch(&mut self, timeout: Duration) -> anyhow::Result<Option<WorkerResult>> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(result) = self.ready.pop_front() {
                return result.map(Some);
            }
            let (_, wakes) = match self.drain_ready() {
                Ok(progress) => progress,
                Err(error) => {
                    // Complete replies remain ahead of a trailing transport or
                    // decoding error in the same drain.
                    self.ready.push_back(Err(error));
                    continue;
                }
            };
            if !self.ready.is_empty() {
                continue;
            }
            self.check_worker("executor poll")?;
            anyhow::ensure!(!wakes.death, "worker rank channel disconnected");
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            self.client
                .wait_wake(remaining.min(WORKER_CHECK_INTERVAL))?;
        }
    }

    /// Drains outstanding calls and asks the rank to shut down.
    ///
    /// A startup that was cancelled goes straight to `terminate`. Otherwise
    /// only a first call acts, and none after `terminate`. A live rank gets up
    /// to `WORKER_DRAIN_TIMEOUT` to answer outstanding batches and is then
    /// sent a shutdown request, without waiting for its answer; `await_close`
    /// completes the retirement. A failure to reach the rank is absorbed, and
    /// `await_close` then bounds the rank by its deadline alone.
    fn request_close(&mut self) {
        // A rank of this launch exited before the group became ready, so this
        // rank is terminated without draining or a shutdown request.
        if self
            .startup_cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(std::sync::atomic::Ordering::Acquire))
        {
            self.terminate();
            return;
        }
        if !matches!(self.retirement, Retirement::Serving) {
            return;
        }

        // Stop the death watcher before the intended teardown, so the exit
        // does not fire a spurious death wake during shutdown.
        let _ = self.death_watcher.take();
        let mut acknowledgment = None;
        // A rank started elsewhere is never observed to have exited here, so
        // shutdown drains its channel as it would a live local rank.
        let exited = matches!(self.child.as_mut().map(Child::try_wait), Some(Ok(Some(_))));
        if !exited {
            // Bound response draining so shutdown can advance to graceful termination
            // and, if necessary, forced process cleanup.
            let drain_deadline = Instant::now() + WORKER_DRAIN_TIMEOUT;
            while !self.pending.is_empty() {
                if Instant::now() >= drain_deadline {
                    tracing::warn!(
                        pending = self.pending.len(),
                        "worker did not drain in-flight responses before shutdown deadline"
                    );
                    break;
                }
                match self.drain_ready() {
                    Ok((0, _)) => {
                        if self.check_worker("shutdown drain").is_err() {
                            break;
                        }
                        let remaining = drain_deadline.saturating_duration_since(Instant::now());
                        let _ = self.client.wait_wake(remaining.min(WORKER_CHECK_INTERVAL));
                    }
                    Ok(_) => {}
                    Err(_) => break,
                }
            }

            // A rank still running after the drain is asked to shut down.
            if !matches!(self.child.as_mut().map(Child::try_wait), Some(Ok(Some(_)))) {
                let message_id = self.alloc_call_id();
                let mut req = WorkerRequest::close();
                req.set_call_id(Some(message_id));
                acknowledgment = self.send_request_checked(&req, "shutdown").ok();
            }
        }
        self.retirement = Retirement::Closing { acknowledgment };
    }

    /// Waits until `deadline` for a rank `request_close` asked to shut down,
    /// then kills its local process if it is still running.
    ///
    /// The shutdown request stays outstanding until the rank answers it, so
    /// the answer, which the rank sends before its teardown begins, always
    /// has a receiver. A local rank is retired once its process exits,
    /// answered or not. The process of a rank elsewhere belongs to its
    /// launcher, so that rank is retired once it answers or its channel
    /// fails; `close_ranks` then follows it to its exit. Acts only on a rank
    /// that is closing.
    fn await_close(&mut self, deadline: Instant) {
        let Retirement::Closing { mut acknowledgment } =
            std::mem::replace(&mut self.retirement, Retirement::Retired)
        else {
            return;
        };

        loop {
            // A failed receive means the rank's end of the channel is gone,
            // which answers the request as finally as a response does.
            if let Some(pending) = acknowledgment.as_ref()
                && !matches!(self.client.try_recv_response(pending), Ok(None))
            {
                acknowledgment = None;
            }
            let retired = match self.child.as_mut() {
                Some(child) => !matches!(child.try_wait(), Ok(None)),
                None => acknowledgment.is_none(),
            };
            if retired {
                return;
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            // A response raises a wake; a process exit does not, so the wait
            // is also bounded by the exit poll interval. A channel that cannot
            // wait any more still gets the interval, so the loop never spins.
            let interval = remaining.min(EXIT_POLL_INTERVAL);
            if self.client.wait_wake(interval).is_err() {
                std::thread::sleep(interval);
            }
        }

        if let Some(child) = self.child.as_mut() {
            tracing::warn!(
                rank = self.rank,
                "worker did not exit before the shutdown deadline; killing it"
            );
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Retires the ranks of one worker group together, absorbing every failure.
///
/// A rank's orderly teardown retires the communicators and process groups it
/// shares with the group's other ranks (`Worker.close` on the Python side),
/// and that retirement completes on any rank only once every rank has
/// reached it. Every rank is therefore asked to shut down, after draining its
/// outstanding calls, before any is waited for, and all of them then share
/// one `WORKER_EXIT_GRACE` deadline, after which a local process still
/// running is killed. Asking one rank at a time would leave the first one
/// waiting on peers that have not been asked, until it is killed with its
/// communicators unreleased.
///
/// `launchers` names the registry that started the group's ranks on other
/// hosts, and the group's worker id. A rank answers its shutdown request
/// before its teardown begins, and a launcher kills the ranks it still runs
/// once the head lets go of the registry, so the ranks elsewhere are held
/// until their launchers report their exits, against the same deadline.
pub(super) fn close_ranks(
    ranks: &mut [RankProcess],
    launchers: Option<(&super::launcher::Launchers, &str)>,
) {
    for rank in ranks.iter_mut() {
        rank.request_close();
    }

    // The ranks elsewhere that were asked to shut down. A rank whose request
    // could not be sent is unreachable, and nothing it could still release
    // waits on this process.
    let remote: BTreeSet<u32> = ranks
        .iter()
        .filter(|rank| {
            rank.child.is_none()
                && matches!(
                    rank.retirement,
                    Retirement::Closing {
                        acknowledgment: Some(_)
                    }
                )
        })
        .map(|rank| rank.rank)
        .collect();

    let deadline = Instant::now() + WORKER_EXIT_GRACE;
    for rank in ranks.iter_mut() {
        rank.await_close(deadline);
    }
    if let Some((registry, worker_id)) = launchers {
        await_remote_exits(registry, worker_id, remote, deadline);
    }
}

/// Waits until `deadline` for the launchers to report the exit of each rank
/// in `remote`, all ranks of `worker_id` on other hosts.
///
/// Gives up when the registry cannot be locked. A rank still running at the
/// deadline is left to its launcher, which kills it once the head releases
/// the registry.
fn await_remote_exits(
    registry: &super::launcher::Launchers,
    worker_id: &str,
    mut remote: BTreeSet<u32>,
    deadline: Instant,
) {
    while !remote.is_empty() {
        let Ok(mut hosts) = super::launcher::lock(registry) else {
            return;
        };
        for (_, exit) in hosts.drain_exits(worker_id) {
            remote.remove(&exit.rank);
        }
        drop(hosts);
        if remote.is_empty() {
            return;
        }

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            tracing::warn!(
                worker = worker_id,
                ranks = ?remote,
                "ranks on other hosts did not exit before the shutdown deadline"
            );
            return;
        }
        std::thread::sleep(remaining.min(EXIT_POLL_INTERVAL));
    }
}

impl Drop for RankProcess {
    /// Closes the rank as `close_ranks` would close it alone. A rank on
    /// another host is not followed to its exit, since no launcher registry
    /// is at hand.
    fn drop(&mut self) {
        self.cancel_startup();
        close_ranks(std::slice::from_mut(self), None);
    }
}

#[cfg(test)]
mod tests {
    use super::{LaneConfig, WorkerProcessArgs, allocator_environment};
    use std::collections::BTreeMap;
    use std::ffi::OsString;

    /// A rank receives the operator's local base checkpoint under the key
    /// the worker reads, and no key when the operator named none, so the
    /// worker then reads the base from the Hugging Face cache.
    #[test]
    fn the_launch_descriptor_carries_a_named_base_checkpoint() {
        let mut args = WorkerProcessArgs {
            model: "/models/FastH3-OmniRef".into(),
            ..WorkerProcessArgs::default()
        };
        let descriptor = args
            .launch_descriptor(0, "127.0.0.1:1", None, None, None)
            .expect("descriptor");
        assert!(descriptor.get("base_model").is_none());

        args.base_model = Some("/models/MiniMax-H3".into());
        let descriptor = args
            .launch_descriptor(0, "127.0.0.1:1", None, None, None)
            .expect("descriptor");
        assert_eq!(descriptor["base_model"], "/models/MiniMax-H3");
    }

    /// `--lane` accepts only the fields the worker applies, so a misspelled
    /// optional limit fails at argument parsing instead of leaving the lane
    /// without that limit.
    #[test]
    fn a_lane_naming_an_unknown_field_is_refused() {
        let lane = r#"{"lane_id":"decode","sm_budget":64,"domains":["decode"]"#;
        let parsed = format!("{lane}}}")
            .parse::<LaneConfig>()
            .expect("decode lane");
        let args = WorkerProcessArgs {
            lanes: vec![parsed],
            ..WorkerProcessArgs::default()
        };
        let descriptor = args
            .launch_descriptor(0, "127.0.0.1:1", None, None, None)
            .expect("descriptor");
        assert_eq!(
            descriptor["lane"][0]["call_kinds"],
            serde_json::json!(["decode", "verify", "token_denoising"])
        );

        for misspelled in [r#""max_batch_token":4096"#, r#""max_inflght":2"#] {
            let parsed = format!("{lane},{misspelled}}}").parse::<LaneConfig>();
            assert!(parsed.is_err(), "{misspelled} must be refused");
        }
    }

    /// A lane partitions compute only: its calls share the worker-wide KV
    /// and latent pools, whose placement the scheduler owns, so a lane
    /// naming a capacity of its own is refused rather than accepted without
    /// effect.
    #[test]
    fn a_lane_naming_a_pool_capacity_is_refused() {
        let lane = r#"{"lane_id":"decode","sm_budget":64,"domains":["decode"]"#;
        for capacity in [
            r#""kv_capacity_tokens":4096"#,
            r#""latent_capacity_units":8"#,
        ] {
            let parsed = format!("{lane},{capacity}}}").parse::<LaneConfig>();
            assert!(parsed.is_err(), "{capacity} must be refused");
        }
    }

    /// Every rank runs with the head's caching-allocator configuration,
    /// under whichever names the head sets, or with expandable segments when
    /// the head configures none, and with its host heap out of transparent
    /// huge pages unless the head decides otherwise. The result names each
    /// variable, since a remote launch carries only the variables its
    /// command sets.
    #[test]
    fn ranks_run_with_the_heads_allocator_configuration() {
        let resolve = |head: &[(&'static str, &str)]| {
            let head: BTreeMap<_, _> = head.iter().copied().collect();
            allocator_environment(|name| head.get(name).map(OsString::from))
                .into_iter()
                .collect::<BTreeMap<_, _>>()
        };
        let expected = |variables: &[(&'static str, &str)]| {
            variables
                .iter()
                .map(|&(name, value)| (name, OsString::from(value)))
                .collect::<BTreeMap<_, _>>()
        };

        assert_eq!(
            resolve(&[]),
            expected(&[
                ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
                ("MIMALLOC_ALLOW_THP", "0"),
            ])
        );
        for head in [
            vec![(
                "PYTORCH_CUDA_ALLOC_CONF",
                "garbage_collection_threshold:0.6",
            )],
            vec![("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")],
            vec![
                ("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync"),
                ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False"),
            ],
        ] {
            let mut ranks = head.clone();
            ranks.push(("MIMALLOC_ALLOW_THP", "0"));
            assert_eq!(resolve(&head), expected(&ranks));
        }
        assert_eq!(
            resolve(&[("MIMALLOC_ALLOW_THP", "1")]),
            expected(&[
                ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
                ("MIMALLOC_ALLOW_THP", "1"),
            ])
        );
    }
}
