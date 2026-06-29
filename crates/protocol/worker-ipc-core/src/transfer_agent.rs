//! Data plane, Tier 2: the pluggable byte mover.
//!
//! [`TransferAgent`] is the minimal one-sided-RDMA-style surface — register a
//! local buffer, resolve a remote segment, submit READ/WRITE requests, poll for
//! completion, and carry an out-of-band notify. It is deliberately addressed in
//! `(addr, len, device)` triples and opaque locators, never tensors or handles:
//! the handle semantics live one tier up in `TensorMover` (the engine's
//! `stage_router`). Backends are selected by a backend string via
//! [`make_transfer_agent`], exactly like TensorRT-LLM's `dlopen`-loaded
//! `BaseTransferAgent` leaves and Mooncake's transfer engine.
//!
//! ## Where the bytes actually move
//!
//! The control plane (this Rust host) never dereferences a tensor: it only routes
//! handles and tracks transfer completion. The **byte movement** happens
//! happens worker↔worker, and because the tensors are torch-owned and addressed
//! by `data_ptr()` integers, the byte-moving Tier-2 backends are realized
//! **worker-side in Python** — `uniserve_worker/runtime/transfer.py`
//! implements `shm` (POSIX shared memory), `cuda_ipc` (CUDA IPC handles via
//! torch reductions), and `mooncake` (one-sided RDMA via Mooncake's
//! `TransferEngine`), all verified moving real tensors cross-process. The host's
//! `TransferAgent` here is the host-side tracking tier: [`InProcessAgent`]
//! models handle registration + the out-of-band notify queue the host watches
//! for write-driven KV completion. [`make_transfer_agent`] accepts every
//! backend name (so `--transfer` validates) and returns the host-side tracking
//! agent; the name is forwarded to the worker, which selects the matching Python
//! transport.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// A registrable local buffer: pinned host DRAM or device VRAM. `device` is the
/// Mooncake-style location label (`"cpu:N"` / `"cuda:N"`) that drives NIC
/// selection in the RDMA backends; the in-process backend ignores it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRegion {
    pub addr: u64,
    pub len: usize,
    pub device: String,
}

/// What [`TransferAgent::register`] returns: the region plus a per-NIC remote
/// key. `register` only pins memory and builds the MR; the buffer stays owned by
/// the caller (register-once + reference-by-(segment, offset)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredRegion {
    pub region: MemoryRegion,
    /// Per-NIC remote key (degenerate `0` for the in-process backend).
    pub rkey: u64,
    /// Local segment id assigned at registration; encoded into locators so a
    /// peer can name this region by `(segment_id, absolute addr)`.
    pub segment_id: u64,
}

/// A resolved remote worker/segment (Mooncake `openSegment` / TRT-LLM
/// `loadRemoteAgent`). Built from a producer's opaque locator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteSegment {
    pub segment_id: u64,
    pub locator: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferOp {
    /// Pull bytes from the remote segment into the local buffer.
    Read,
    /// Push bytes from the local buffer to the remote segment.
    Write,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalAddr {
    pub addr: u64,
    pub len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteAddr {
    pub segment_id: u64,
    pub addr: u64,
}

/// One async one-sided transfer request. READ pulls remote→local, WRITE pushes
/// local→remote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferReq {
    pub op: TransferOp,
    pub src: LocalAddr,
    pub dst: RemoteAddr,
    pub len: usize,
}

/// Opaque handle to a submitted batch of transfers; poll it for completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransferTicket(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStatus {
    Pending,
    Completed,
    Failed,
}

/// Tier-2 one-sided byte mover. All addressing is `(addr, len, device)` and
/// opaque locators — never tensors. See module docs.
pub trait TransferAgent: Send + Sync {
    /// Pin a local buffer + build its MR; returns a remotely-addressable
    /// description. Mooncake `registerLocalMemory` / TRT-LLM `registerMemory`.
    fn register(&self, region: MemoryRegion) -> RegisteredRegion;

    /// Release a previously registered region.
    fn deregister(&self, region: &RegisteredRegion);

    /// Resolve a remote segment from a producer's opaque locator. Mooncake
    /// `openSegment` / TRT-LLM `loadRemoteAgent`.
    fn open_remote(&self, locator: &[u8]) -> RemoteSegment;

    /// Submit a batch of async one-sided transfers. Mooncake `submitTransfer` /
    /// TRT-LLM `submitTransferRequests`.
    fn submit(&self, reqs: Vec<TransferReq>) -> TransferTicket;

    /// Poll a ticket for completion (poll model, not callbacks). Mooncake
    /// `getBatchTransferStatus`.
    fn poll(&self, ticket: &TransferTicket) -> TransferStatus;

    /// Out-of-band notify: hand the peer a message once transfers land, so no
    /// separate control RPC is needed. Mooncake `submitTransferWithNotify` /
    /// TRT-LLM `notifySyncMessage`.
    fn notify(&self, remote: &RemoteSegment, msg: Vec<u8>);

    /// Drain notifies addressed to this agent.
    fn drain_notifies(&self) -> Vec<Vec<u8>>;
}

/// Backend configuration passed to [`make_transfer_agent`]. Backend-specific
/// keys (NIC lists, Mooncake metadata server, …) ride in `options`.
#[derive(Debug, Clone, Default)]
pub struct AgentConfig {
    /// This agent's own segment/connection label (e.g. `"cuda:0"`).
    pub local_segment: String,
    pub options: std::collections::BTreeMap<String, String>,
}

/// Backend names whose byte transport is realized worker-side in Python
/// (`runtime/transfer.py`). The host only tracks handles for these.
pub const WORKER_SIDE_BACKENDS: &[&str] = &["shm", "cuda_ipc", "mooncake"];

/// Select the **host-side** Tier-2 agent by backend name (TRT-LLM's
/// `dlopen`-by-backend-string model). `inproc`/`local` is the in-process
/// tracker; `shm`/`cuda_ipc`/`mooncake` also return the host-side tracker
/// ([`InProcessAgent`]) because the host never moves tensors — the real byte
/// real byte transport for those edges runs worker-side in Python (see module
/// docs). Unknown names error so a `--transfer` typo fails loudly.
pub fn make_transfer_agent(
    backend: &str,
    _cfg: &AgentConfig,
) -> anyhow::Result<Box<dyn TransferAgent>> {
    match backend {
        "inproc" | "local" => Ok(Box::new(InProcessAgent::new())),
        b if WORKER_SIDE_BACKENDS.contains(&b) => Ok(Box::new(InProcessAgent::new())),
        other => anyhow::bail!("unknown transfer backend {other:?}"),
    }
}

/// In-process backend: same-process, zero byte movement.
///
/// The in-process data plane is direct-reference — producer and consumer share
/// an address space and a handle already names a live buffer, so there is
/// nothing to copy. `submit` therefore records a ticket that is immediately
/// `Completed`, `register`/`open_remote` only mint ids, and `notify` is a local
/// FIFO. This is a faithful `TransferAgent` (so the `TensorMover` seam and the
/// read-driven-fetch flow are exercised) that happens to move no bytes.
#[derive(Debug, Default)]
pub struct InProcessAgent {
    next_segment: AtomicU64,
    next_ticket: AtomicU64,
    notifies: Mutex<VecDeque<Vec<u8>>>,
}

impl InProcessAgent {
    pub fn new() -> Self {
        Self {
            next_segment: AtomicU64::new(1),
            next_ticket: AtomicU64::new(1),
            notifies: Mutex::new(VecDeque::new()),
        }
    }
}

impl TransferAgent for InProcessAgent {
    fn register(&self, region: MemoryRegion) -> RegisteredRegion {
        RegisteredRegion {
            region,
            rkey: 0,
            segment_id: self.next_segment.fetch_add(1, Ordering::Relaxed),
        }
    }

    fn deregister(&self, _region: &RegisteredRegion) {}

    fn open_remote(&self, locator: &[u8]) -> RemoteSegment {
        // The in-process locator is the 8-byte little-endian segment id (or
        // empty for a degenerate worker-local handle).
        let segment_id = if locator.len() == 8 {
            u64::from_le_bytes(locator.try_into().expect("checked len == 8"))
        } else {
            0
        };
        RemoteSegment {
            segment_id,
            locator: locator.to_vec(),
        }
    }

    fn submit(&self, _reqs: Vec<TransferReq>) -> TransferTicket {
        // Zero-transfer: the ticket is born complete.
        TransferTicket(self.next_ticket.fetch_add(1, Ordering::Relaxed))
    }

    fn poll(&self, _ticket: &TransferTicket) -> TransferStatus {
        TransferStatus::Completed
    }

    fn notify(&self, _remote: &RemoteSegment, msg: Vec<u8>) {
        self.notifies
            .lock()
            .expect("notify queue poisoned")
            .push_back(msg);
    }

    fn drain_notifies(&self) -> Vec<Vec<u8>> {
        self.notifies
            .lock()
            .expect("notify queue poisoned")
            .drain(..)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inproc_agent_completes_immediately_and_carries_notifies() {
        let agent = make_transfer_agent("inproc", &AgentConfig::default()).unwrap();
        let reg = agent.register(MemoryRegion {
            addr: 0x1000,
            len: 256,
            device: "cpu:0".into(),
        });
        assert_eq!(reg.rkey, 0);
        assert!(reg.segment_id >= 1);

        let ticket = agent.submit(vec![TransferReq {
            op: TransferOp::Read,
            src: LocalAddr {
                addr: 0x1000,
                len: 256,
            },
            dst: RemoteAddr {
                segment_id: reg.segment_id,
                addr: 0x2000,
            },
            len: 256,
        }]);
        assert_eq!(agent.poll(&ticket), TransferStatus::Completed);

        let remote = agent.open_remote(&reg.segment_id.to_le_bytes());
        assert_eq!(remote.segment_id, reg.segment_id);
        agent.notify(&remote, b"kv-ready".to_vec());
        assert_eq!(agent.drain_notifies(), vec![b"kv-ready".to_vec()]);
        assert!(agent.drain_notifies().is_empty());
    }

    #[test]
    fn factory_accepts_known_backends_and_rejects_unknown() {
        // Every known backend resolves to a host-side tracking agent (the real
        // shm/cuda_ipc/mooncake byte transport runs worker-side in Python).
        for backend in ["inproc", "local", "shm", "cuda_ipc", "mooncake"] {
            assert!(
                make_transfer_agent(backend, &AgentConfig::default()).is_ok(),
                "backend {backend:?} should resolve"
            );
        }
        assert!(make_transfer_agent("nope", &AgentConfig::default()).is_err());
    }
}
