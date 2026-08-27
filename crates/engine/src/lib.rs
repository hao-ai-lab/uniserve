//! In-process engine runtime: scheduler ownership, executor construction,
//! health, stats, control surface, and shutdown.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod core;
pub mod executor;
mod handle;
pub mod kv;
pub mod scheduler;
pub mod sim;
pub mod worker;

pub use crate::core::{EngineBackend, EngineCore, EngineCoreConfig};
pub use crate::executor::{TransferBackend, TransferSpec, WorkersSpec};
pub use crate::handle::{
    Command, EVENT_BUFFER_CAPACITY, EngineHandle, EventRx, EventSendError, EventTx, MediaEventRx,
    MediaEventSendError, MediaEventTx, StreamCancelCause, SubmitError,
};
pub use crate::scheduler::{
    DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS, SchedulingPolicy,
};
pub use crate::worker::{
    FlashInferBackend, MultiprocExecutor, StagedExecutor, UniprocExecutor, WorkerLaunchConfig,
    WorkerSpawnSpec,
};
pub use uniserve_worker_ipc::AttentionBackend;
