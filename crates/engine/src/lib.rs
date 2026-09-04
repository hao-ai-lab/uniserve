//! In-process scheduling, memory management, worker execution, and request control.
//!
//! The crate owns the engine thread and exports request handles, configuration,
//! execution backends, statistics, and a GPU-free simulator.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod core;
mod executor;
mod handle;
mod kv;
/// Scheduler-owned allocation and worker-placement primitives.
pub mod memory;
mod runtime;
mod scheduler;
mod sim;
mod worker;

pub use crate::core::{EngineCore, EngineCoreConfig};
pub use crate::executor::{
    Batch, BatchResult, Executor, ExecutorError, ExecutorInfo, ExecutorSubmitError, Op,
    OpPlacement, PhysicalExecutor, PhysicalSubmitError, PoolConfig, PoolId, TransferBackend,
    TransferEdge, TransportMap, TransportMapError, WorkerExecError, WorkerLossError,
    WorkerTopology, WorkerTopologyError,
};
pub use crate::handle::{
    Command, EVENT_BUFFER_CAPACITY, EngineHandle, EventRx, EventSendError, EventTx,
    StreamCancelCause, SubmitError,
};
pub use crate::runtime::{
    ArRuntime, ControlTokens, DiffusionRuntime, EngineLoop, Runtime, RuntimeProfile, UmmRuntime,
};
pub use crate::scheduler::{
    DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS, DomainStats, EncoderStats,
    ExecutionDomainStats, GeneralStats, KvCacheStats, PrefixStats, SchedStats, SchedStatsReporter,
    Scheduler, SchedulerConfig, SchedulingPolicy, TimingStats, WorkerStats,
};
pub use crate::sim::{SimEngine, SimExecutor};
pub use crate::worker::{
    FlashInferBackend, FlashInferBackendParseError, LaneConfig, MultiprocExecutor, StagedExecutor,
    UniprocExecutor, WorkerProcessArgs,
};
pub use uniserve_worker_ipc::AttentionBackend;
