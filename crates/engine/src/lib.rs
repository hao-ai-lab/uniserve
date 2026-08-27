//! In-process engine runtime: scheduler ownership, executor construction,
//! stats, control surface, and shutdown.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod core;
mod executor;
mod handle;
mod kv;
mod scheduler;
mod sim;
mod worker;

pub use crate::core::{EngineCore, EngineCoreConfig};
pub use crate::executor::{
    ControlAck, ControlOp, Executor, Pool, TransferBackend, TransportMap, TransportMapError,
    WorkerExecError, WorkerKind, WorkerLossError, WorkerTopology, WorkerTopologyError,
};
pub use crate::handle::{
    Command, EVENT_BUFFER_CAPACITY, EngineHandle, EventRx, EventSendError, EventTx, MediaEventRx,
    MediaEventSendError, MediaEventTx, StreamCancelCause, SubmitError,
};
pub use crate::scheduler::{
    ControlTokens, DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH,
    DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS,
    DomainStats, EncoderStats, ExecutionDomainStats, GeneralStats, KvCacheStats, PrefixStats,
    SchedStats, SchedStatsReporter, Scheduler, SchedulingPolicy, TimingStats, WorkerStats,
};
pub use crate::sim::{SimEngine, SimExecutor};
pub use crate::worker::{
    FlashInferBackend, FlashInferBackendParseError, LaneConfig, MultiprocExecutor, StagedExecutor,
    UniprocExecutor, WorkerProcessArgs,
};
pub use uniserve_worker_ipc::AttentionBackend;
