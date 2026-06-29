//! In-process engine runtime: scheduler ownership, executor construction,
//! health, stats, control surface, and shutdown.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod core;

pub use crate::core::{
    EngineBackend, EngineCore, EngineCoreBuilder, EngineCoreConfig, NeedsConfig, NeedsExecutor,
    Ready,
};
pub use uniserve_scheduler::{
    DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS, SchedulingPolicy,
};
