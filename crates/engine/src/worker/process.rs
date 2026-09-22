//! Single-worker executor over an iceoryx2 request-response service.
//!
//! The host exchanges FlatBuffers descriptors and bounded result values while
//! tensors, KV pages, and latent storage remain worker-resident.

use crate::executor::WorkerResult;
use std::collections::{HashMap, HashSet, VecDeque};
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
/// already-bound connection rather than waiting for the rank to start.
const CHANNEL_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

use crate::worker::WorkerProcessArgs;
use crate::worker::death_watch::DeathWatcher;

/// Deadline for the initial worker connect / info handshake, where the worker may still
/// be loading a large model and the IPC server may not yet be connected.
const WORKER_CONNECT_TIMEOUT: Duration = Duration::from_secs(300);
/// Per-call backpressure deadline for steady-state sends (batch submit / control). The
/// IPC server is already connected by this point, so a missing server connection means
/// the worker has dropped off and we should fail fast rather than block for the full
/// startup grace period.
const WORKER_SEND_TIMEOUT: Duration = Duration::from_secs(30);
/// Maximum time `shutdown` will spend draining in-flight responses from a still-alive
/// worker before falling through to the graceful-shutdown request and kill fallback. A
/// hung (alive but unresponsive) worker must not be able to block shutdown forever.
const WORKER_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);
const STARTUP_LOG_INTERVAL: Duration = Duration::from_secs(30);
const WORKER_CHECK_INTERVAL: Duration = Duration::from_millis(500);

/// One configured model-execution lane passed in [`WorkerProcessArgs`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneConfig {
    /// Stable lane identity within the worker process.
    pub lane_id: String,
    /// Streaming-multiprocessor budget assigned to the lane.
    pub sm_budget: u32,
    /// Public JSON capability selectors resolved to call kinds by worker startup.
    pub domains: Vec<String>,
    /// Optional lane-local KV capacity in tokens.
    pub kv_capacity_tokens: Option<u64>,
    /// Optional lane-local latent capacity in allocation units.
    pub latent_capacity_units: Option<u64>,
    /// Optional call-count limit per batch.
    pub max_batch_calls: Option<u32>,
    /// Optional token-count limit per batch.
    pub max_batch_tokens: Option<u32>,
    /// Optional unresolved-run limit.
    pub max_inflight: Option<u32>,
}

impl std::str::FromStr for LaneConfig {
    type Err = String;

    /// Parses the value from its string representation.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let lane: Self = serde_json::from_str(value)
            .map_err(|error| format!("invalid execution lane JSON: {error}"))?;
        if lane.lane_id.is_empty()
            || lane.sm_budget == 0
            || lane.domains.is_empty()
            || lane
                .domains
                .iter()
                .any(|name| !matches!(name.as_str(), "prefill" | "decode" | "flow"))
            || lane.domains.iter().collect::<HashSet<_>>().len() != lane.domains.len()
        {
            return Err("execution lane identity, SM budget, and domains must be valid".into());
        }
        Ok(lane)
    }
}

impl LaneConfig {
    /// Serializes the lane as the worker command-line JSON value.
    pub fn worker_arg(&self) -> String {
        serde_json::json!({
            "lane_id": self.lane_id,
            "sm_budget": self.sm_budget,
            "domains": self.domains,
            "kv_capacity_tokens": self.kv_capacity_tokens,
            "latent_capacity_units": self.latent_capacity_units,
            "max_batch_calls": self.max_batch_calls,
            "max_batch_tokens": self.max_batch_tokens,
            "max_inflight": self.max_inflight,
        })
        .to_string()
    }
}

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

    /// Parses the value from its string representation.
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
            checkpoint_identity: None,
            ranks: vec![crate::WorkerRank {
                node: "localhost".into(),
                device: "cuda:0".into(),
            }],
            host: "localhost".into(),
            // One local rank running one component. A launch replaces this
            // with the components the model being served declared.
            components: crate::WorkerConfig::single_component(DEFAULT_COMPONENT, 1),
            peers: Default::default(),
            stub: false,
            queue_depth: 2,
            req_slot_cap: 1 << 20,
            resp_slot_cap: 8 << 20,
            kv_token_capacity: None,
            block_size: 64,
            max_batch_calls: 128,
            max_batch_tokens: 16_384,
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
            kv_memory_fraction: 0.70,
            mesh: None,
            distributed_backend: None,
            lanes: Vec::new(),
            graph_policy: "auto".into(),
            decode_graph_batch_sizes: None,
            prefill_cuda_graph: false,
            prefill_graph_token_sizes: None,
            flow_graph_batch_sizes: None,
            flow_graph_shapes: None,
            video_graph_shapes: None,
            flashinfer_workspace_size: 512 * 1024 * 1024,
            flashinfer_use_tensor_core: None,
            flashinfer_decode_backend: FlashInferBackend::Fa2,
            flashinfer_prefill_backend: FlashInferBackend::Auto,
            flashinfer_decode_split_tile_size: None,
            flashinfer_prefill_split_tile_size: None,
            flashinfer_disable_split_kv: false,
            max_model_len: 8192,
            max_video_seconds: 15.0,
        }
    }
}

impl WorkerProcessArgs {
    /// Derives the checkpoint identity once when the model is a local
    /// directory the head can read.
    ///
    /// A Hub identifier or a model this host cannot read yields no
    /// expectation; the ranks are then held to each other's report instead.
    /// An identity already stated by the caller is kept.
    pub(super) fn derive_checkpoint_identity(&mut self) -> anyhow::Result<()> {
        if self.checkpoint_identity.is_some() || self.stub {
            return Ok(());
        }
        let root = std::path::Path::new(&self.model);
        if root.is_dir() {
            self.checkpoint_identity = Some(super::checkpoint::checkpoint_identity(root)?);
        }
        Ok(())
    }

    /// Returns one rank's device index on its own host and that host's rank count.
    ///
    /// Section 5.1 of the serving architecture gives `LOCAL_RANK` and
    /// `LOCAL_WORLD_SIZE` host-relative values: a rank's position among the
    /// ranks the placement puts on the same node, and how many ranks that node
    /// holds. Host count enters the system here and nowhere below it. The two
    /// coincide with the global values while an instance occupies one host.
    fn host_slot(&self, rank: u32) -> (u32, u32) {
        let node = &self.ranks[rank as usize].node;
        let resident = self
            .ranks
            .iter()
            .enumerate()
            .filter(|(_, placed)| &placed.node == node)
            .map(|(position, _)| position)
            .collect::<Vec<_>>();
        let index = resident
            .iter()
            .position(|position| *position == rank as usize)
            .expect("a rank is resident on the node its placement names");
        (
            u32::try_from(index).expect("a host's rank index fits the launch arithmetic"),
            u32::try_from(resident.len()).expect("a host's rank count fits the launch arithmetic"),
        )
    }

    /// Builds the typed launch descriptor the worker process consumes.
    ///
    /// Every tuning value is stated explicitly, so the launching side is the
    /// single source of defaults and the worker never re-derives one. The keys
    /// are the worker configuration's own field names; only the process
    /// identity and endpoint travel on argv.
    fn launch_descriptor(
        &self,
        device: &str,
        rank: u32,
        world_size: u32,
        registration: &str,
        components: &std::collections::BTreeMap<String, crate::executor::ComponentConfig>,
        transfer_backends: &str,
        publish_backends: &str,
        distributed_init_method: Option<String>,
        channel_transport: &str,
        acknowledgment_slot: u32,
        host_slots: &[u32],
        products_cross_hosts: bool,
    ) -> anyhow::Result<serde_json::Value> {
        let depth = self.queue_depth.max(1);
        let max_payload = self.req_slot_cap.max(self.resp_slot_cap).max(1);
        let mut fields = serde_json::Map::new();
        fields.insert("worker_id".into(), json!(self.worker_id));
        fields.insert("registration_address".into(), json!(registration));
        // The placement decides the mechanism and the rank names the endpoint:
        // a rank on the head's host can offer shared memory, a rank elsewhere
        // cannot, and only the head knows where a rank was placed.
        fields.insert("channel_transport".into(), json!(channel_transport));
        // A consumer writes its own slot's word in every chunk or segment it
        // reads; the slots a producer watches travel on each producing call.
        fields.insert("acknowledgment_slot".into(), json!(acknowledgment_slot));
        // Which of those slots are on this rank's host decides the mechanism
        // a host product is published over: shared memory reaches the host,
        // the rank channel reaches the rest.
        fields.insert("host_slots".into(), json!(host_slots));
        // Readiness is a producer synchronize only where an interprocess event
        // cannot carry it, which is exactly where a consumer is on another
        // host. Only the placement knows that.
        fields.insert("products_cross_hosts".into(), json!(products_cross_hosts));
        fields.insert("queue_depth".into(), json!(depth));
        fields.insert("ipc_payload_cap".into(), json!(max_payload));
        fields.insert("model".into(), json!(self.model));
        // The expectation travels only when the head could derive one; a
        // rank then refuses a checkpoint whose identity differs from it.
        if let Some(identity) = &self.checkpoint_identity {
            fields.insert("checkpoint_identity".into(), json!(identity));
        }
        fields.insert("device".into(), json!(device));
        fields.insert("rank".into(), json!(rank));
        fields.insert("local_rank".into(), json!(self.host_slot(rank).0));
        fields.insert("world_size".into(), json!(world_size));
        fields.insert("components".into(), serde_json::to_value(components)?);
        fields.insert(
            "supported_calls".into(),
            if self.capability_groups.is_empty() {
                Value::Null
            } else {
                json!(self.capability_groups.join(","))
            },
        );
        fields.insert("transfer_backends".into(), json!(transfer_backends));
        fields.insert("publish_backends".into(), json!(publish_backends));
        fields.insert(
            "distributed_init_method".into(),
            json!(distributed_init_method),
        );
        fields.insert(
            "distributed_backend".into(),
            json!(self.distributed_backend),
        );
        fields.insert("mesh".into(), json!(self.mesh));
        fields.insert(
            "lane".into(),
            json!(
                self.lanes
                    .iter()
                    .map(|lane| lane.worker_arg())
                    .collect::<Vec<_>>()
            ),
        );
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
        fields.insert("kv_memory_fraction".into(), json!(self.kv_memory_fraction));
        fields.insert("kv_token_capacity".into(), json!(self.kv_token_capacity));
        fields.insert(
            "attention_backend".into(),
            json!(self.attention_backend.as_name()),
        );
        fields.insert("block_size".into(), json!(self.block_size));
        fields.insert("max_batch_calls".into(), json!(self.max_batch_calls));
        fields.insert("max_batch_tokens".into(), json!(self.max_batch_tokens));
        fields.insert("max_model_len".into(), json!(self.max_model_len));
        fields.insert("max_video_seconds".into(), json!(self.max_video_seconds));
        fields.insert("graph_policy".into(), json!(self.graph_policy));
        fields.insert(
            "decode_graph_batch_sizes".into(),
            json!(self.decode_graph_batch_sizes),
        );
        fields.insert("prefill_cuda_graph".into(), json!(self.prefill_cuda_graph));
        fields.insert(
            "prefill_graph_token_sizes".into(),
            json!(self.prefill_graph_token_sizes),
        );
        fields.insert(
            "flow_graph_batch_sizes".into(),
            json!(self.flow_graph_batch_sizes),
        );
        fields.insert("flow_graph_shapes".into(), json!(self.flow_graph_shapes));
        fields.insert("video_graph_shapes".into(), json!(self.video_graph_shapes));
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

/// Single-process worker executor over iceoryx2 IPC.
pub(super) struct RankProcess {
    client: RankChannel,
    info: WorkerInfo,
    startup_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// The rank's process, when this engine started it. A rank placed on
    /// another host is started by that host's launcher and its process is
    /// never visible here; its liveness is its channel connection, which the
    /// launcher closes when it terminates the rank.
    child: Option<Child>,
    /// The collective rendezvous address this group's ranks share.
    _rendezvous: Option<String>,
    /// Retains the launch descriptor until the worker has read it.
    _launch_descriptor: tempfile::TempDir,
    depth: usize,
    rank: u32,
    world_size: u32,
    expected_components: std::collections::BTreeMap<String, uniserve_core::ComponentConfig>,
    pending: HashMap<u64, PendingRecord>,
    ready: VecDeque<anyhow::Result<WorkerResult>>,
    next_message_id: u64,
    shutdown_sent: bool,
    /// Edge-triggered worker-death watcher: fires the scheduler park's death
    /// wake when the child exits. `None` when polling or when `pidfd` could not
    /// be opened (falls back to the bounded liveness probe).
    death_watcher: Option<DeathWatcher>,
}

/// One spawned rank between process creation and its endpoint report.
///
/// The engine cannot bind the rank's channel until the rank names its own
/// endpoint, so everything the channel needs is retained here meanwhile.
pub(super) struct PendingRank {
    /// Taken by adoption; a rank still held here is killed when the launch fails.
    child: Option<Child>,
    rank: u32,
    world_size: u32,
    depth: usize,
    max_payload: usize,
    /// Taken by adoption alongside the process it cancels.
    startup_abort: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    rendezvous: Option<String>,
    /// Retains the launch descriptor until the rank has read it; adoption
    /// transfers it to the channel that outlives this launch phase.
    launch_descriptor: Option<tempfile::TempDir>,
    components: std::collections::BTreeMap<String, crate::executor::ComponentConfig>,
}

struct PendingRecord {
    kind: OutstandingKind,
    pending: Outstanding,
}

/// Releases the consumed request.
fn release_consumed_request(record: PendingRecord) -> OutstandingKind {
    let PendingRecord { kind, pending } = record;
    drop(pending);
    kind
}

enum OutstandingKind {
    Batch { batch_id: u64 },
}

impl PendingRank {
    /// Spawns one rank, which reports its endpoint to `registration`.
    ///
    /// The rank's channel is bound afterwards from that report, so nothing here
    /// names the endpoint and nothing waits for model readiness.
    pub(crate) fn spawn_rank(
        args: &WorkerProcessArgs,
        device: &str,
        rank: u32,
        world_size: u32,
        rendezvous: Option<String>,
        channel_transport: &str,
        remote: Option<&mut super::launcher::RemoteHost<'_>>,
        components: &std::collections::BTreeMap<String, crate::executor::ComponentConfig>,
        startup_abort: std::sync::Arc<std::sync::atomic::AtomicBool>,
        registration: &str,
    ) -> anyhow::Result<Self> {
        let depth = args.queue_depth.max(1);
        let max_payload = args.req_slot_cap.max(args.resp_slot_cap).max(1);

        // Resolve mechanism ownership from the physical rank's incident edges.
        let (backends, publications) = args.transfer.rank_backends(&args.worker_id, rank);
        // A rank cannot name the ranks that read what it publishes: it knows
        // its own component, not which component consumes its products, and a
        // product is consumed in a later batch than the one producing it. The
        // head states a product's readers on the call that produces it and
        // gives the rank its own slot here.
        let acknowledgment_slot = args.transfer.acknowledgment_slot(&args.worker_id, rank);
        let products_cross_hosts = args.transfer.products_cross_hosts(&args.worker_id, rank);
        let host_slots = args.transfer.host_slots(&args.worker_id, rank);
        let names = |backends: &std::collections::BTreeSet<crate::executor::TransferBackend>| {
            backends
                .iter()
                .map(|backend| backend.as_str())
                .collect::<Vec<_>>()
                .join(",")
        };
        let distributed_init_method = rendezvous.clone();

        // One typed descriptor carries every launch value. argv keeps the
        // process identity and the descriptor's location so a running worker
        // remains identifiable from the process table.
        let descriptor = args.launch_descriptor(
            device,
            rank,
            world_size,
            registration,
            components,
            &names(&backends),
            &names(&publications),
            distributed_init_method,
            channel_transport,
            acknowledgment_slot,
            &host_slots,
            products_cross_hosts,
        )?;
        let descriptor_directory = tempfile::Builder::new()
            .prefix("uniserve-worker-launch")
            .tempdir()
            .context("creating the worker launch descriptor directory")?;
        let descriptor_path = descriptor_directory.path().join("launch.json");
        std::fs::write(&descriptor_path, serde_json::to_vec_pretty(&descriptor)?)
            .context("writing the worker launch descriptor")?;

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
        let (local_rank, local_world_size) = args.host_slot(rank);
        cmd.env("RANK", rank.to_string())
            .env("WORLD_SIZE", world_size.to_string())
            .env("LOCAL_RANK", local_rank.to_string())
            .env("LOCAL_WORLD_SIZE", local_world_size.to_string());
        // Every rank serves varying shapes from expandable allocator segments.
        // Publication never depends on the caching allocator: a device product
        // is exported from the rank's own VMM arena or copied into its bounded
        // VMM pool, both reserved outside the allocator.
        if std::env::var_os("PYTORCH_ALLOC_CONF").is_none()
            && std::env::var_os("PYTORCH_CUDA_ALLOC_CONF").is_none()
        {
            cmd.env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True");
        }
        if let Ok(cwd) = std::env::current_dir() {
            let pp = std::env::var("PYTHONPATH").unwrap_or_default();
            cmd.env("PYTHONPATH", format!("{}:{}", cwd.display(), pp));
        }
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
                child: None,
                rank,
                world_size,
                depth,
                max_payload,
                startup_abort: Some(startup_abort),
                rendezvous,
                launch_descriptor: Some(descriptor_directory),
                components: components.clone(),
            });
        }
        let child = cmd.spawn().context("spawning python worker")?;

        Ok(Self {
            child: Some(child),
            rank,
            world_size,
            depth,
            max_payload,
            startup_abort: Some(startup_abort),
            rendezvous,
            launch_descriptor: Some(descriptor_directory),
            components: components.clone(),
        })
    }

    /// Returns the rank this process was launched as.
    pub(crate) fn rank(&self) -> u32 {
        self.rank
    }

    /// Fails by name when the rank exited before reporting its endpoint.
    pub(crate) fn check_alive(&mut self) -> anyhow::Result<()> {
        // A rank started by another host's launcher has no process here. Its
        // liveness is its connection, as section 5.1 states, and its exit
        // reaches the head as a report from the launcher that owns it.
        let Some(child) = self.child.as_mut() else {
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
    pub(crate) fn adopt(mut self, transport: &str, endpoint: &str) -> anyhow::Result<RankProcess> {
        let rank = self.rank;
        let world_size = self.world_size;
        let depth = self.depth;
        // The process stays owned here until the channel exists, so a failed
        // connection terminates the rank instead of orphaning it.
        let client = RankChannel::connect(
            transport,
            endpoint,
            self.max_payload,
            depth,
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
        let child = self.child.take();
        let launch_descriptor = self
            .launch_descriptor
            .take()
            .expect("an unadopted rank retains its launch descriptor");
        let startup_abort = self
            .startup_abort
            .take()
            .expect("an unadopted rank retains its startup cancellation");
        let rendezvous = self.rendezvous.take();
        let components = std::mem::take(&mut self.components);
        // A watcher observes a process identifier, so only a rank this engine
        // started has one. A rank elsewhere falls to the bounded liveness
        // probe, which reads its channel rather than its process.
        let death_watcher = child.as_ref().and_then(|child| {
            DeathWatcher::spawn(child.id(), client.death_wake(), startup_abort.clone())
        });
        Ok(RankProcess {
            client,
            info: WorkerInfo::default(),
            startup_cancel: Some(startup_abort),
            child,
            _rendezvous: rendezvous,
            _launch_descriptor: launch_descriptor,
            depth,
            rank,
            world_size,
            expected_components: components,
            pending: HashMap::new(),
            ready: VecDeque::new(),
            next_message_id: 1,
            shutdown_sent: false,
            death_watcher,
        })
    }
}

impl Drop for PendingRank {
    /// A rank that never reported an endpoint has no channel to close through,
    /// so a failed launch terminates its process here rather than leaking it.
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl RankProcess {
    /// Publish capabilities only after model resources and warmup are ready.
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

    /// Cancel unfinished rank startup when a required peer exits.
    pub(crate) fn set_startup_cancel(
        &mut self,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) {
        self.startup_cancel = cancel;
    }

    /// Terminates a failed or cancelled rank and waits for process-owned resources to retire.
    pub(crate) fn terminate(&mut self) {
        self.shutdown_sent = true;
        self.death_watcher.take();
        // A rank this engine started is terminated here. A rank elsewhere is
        // terminated by its launcher when this head's connection closes, which
        // is the liveness contract the launcher was given.
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.pending.clear();
        self.ready.clear();
    }

    /// Checks child liveness and cancellation of an unfinished startup.
    pub(super) fn check_worker(&mut self, context: &str) -> anyhow::Result<()> {
        if self
            .startup_cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(std::sync::atomic::Ordering::Acquire))
        {
            bail!("worker startup cancelled during {context}");
        }
        if let Some(child) = self.child.as_mut() {
            if let Some(status) = child.try_wait()? {
                bail!("worker process exited during {context}: {status}");
            }
        }
        Ok(())
    }

    /// Sends the request checked.
    fn send_request_checked(
        &mut self,
        req: &WorkerRequest,
        context: &str,
    ) -> anyhow::Result<Outstanding> {
        self.send_request_with_timeout(req, context, WORKER_SEND_TIMEOUT)
    }

    /// Waits for an IPC server connection and submits a request before the deadline.
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
        loop {
            if let Some(frame) = self.client.try_recv_response(pending)? {
                return Ok(frame);
            }
            anyhow::ensure!(
                started.elapsed() < Duration::from_secs(300),
                "worker response timed out during {context}"
            );
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

    /// Drains ready IPC responses and routes them to physical-run completion state.
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
            // Consuming the response ends this IPC request. Release its
            // iceoryx active-request slot before routing the result.
            let kind = release_consumed_request(record);
            self.route(message_id, kind, frame)?;
            drained += 1;
        }
        Ok((drained, wakes))
    }

    /// Acquires output storage before validating response correlation.
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
    /// A shared-memory channel multiplexes every wake onto one listener; a
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

    /// Submits one physical run and records its outstanding call identities.
    pub(super) fn submit_batch(&mut self, batch: Batch) -> Result<(), BatchSubmitError> {
        self.drain_ready().map_err(BatchSubmitError::Failed)?;
        if self.pending.len() >= self.depth {
            return Err(BatchSubmitError::WouldBlock(batch));
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

    /// Drains outstanding calls, requests graceful shutdown, and bounds forced termination.
    pub(super) fn close(&mut self) -> anyhow::Result<()> {
        if self
            .startup_cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(std::sync::atomic::Ordering::Acquire))
        {
            self.terminate();
            return Ok(());
        }
        if self.shutdown_sent {
            return Ok(());
        }
        self.shutdown_sent = true;
        // Stop the death watcher before we intentionally tear the worker down,
        // so its exit does not fire a spurious death wake during shutdown.
        let _ = self.death_watcher.take();
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
            if !matches!(self.child.as_mut().map(Child::try_wait), Some(Ok(Some(_)))) {
                let message_id = self.alloc_call_id();
                let mut req = WorkerRequest::close();
                req.set_call_id(Some(message_id));
                if let Ok(pending) = self.send_request_checked(&req, "shutdown") {
                    let _ = self
                        .client
                        .recv_response_timeout(&pending, Duration::from_secs(5));
                }
            }
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while let Some(child) = self.child.as_mut() {
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(100));
                }
                _ => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break;
                }
            }
        }
        Ok(())
    }
}

impl Drop for RankProcess {
    /// Releases resources owned by this value.
    fn drop(&mut self) {
        let _ = self.close();
    }
}
