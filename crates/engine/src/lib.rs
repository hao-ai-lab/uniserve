//! In-process engine runtime: scheduler ownership, executor construction,
//! health, stats, control surface, and shutdown.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod core;
pub mod executor;
mod handle;
pub mod kv;
pub mod process;
pub mod scheduler;
pub mod sim;
pub mod worker;

pub use crate::core::{
    EngineBackend, EngineCore, EngineCoreBuilder, EngineCoreConfig, NeedsConfig, NeedsExecutor,
    Ready,
};
pub use crate::handle::{
    Command, EVENT_BUFFER_CAPACITY, EngineHandle, EventRx, EventSendError, EventTx, MediaEventRx,
    MediaEventTx,
};
pub use crate::scheduler::{
    DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS, SchedulingPolicy,
};
pub use crate::worker::{MultiprocExecutor, StageRouter, UniprocExecutor, WorkerLaunchConfig};
