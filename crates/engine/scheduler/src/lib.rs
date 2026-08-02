//! Stateful scheduling, admission, batching, and request control.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub(crate) mod bench_trace;
pub(crate) mod cpu_continuation;
pub mod generation;
pub mod grammar;
pub(crate) mod image_artifact;
pub mod logits;
pub mod policy;
pub(crate) mod prefix_cache;
pub mod queue;
pub mod resources;
pub mod scheduler;
pub(crate) mod spec_decode;
pub mod stats_report;
pub mod trace;

pub use grammar::{CompiledGrammar, GrammarCompiler, GrammarMatcher};
pub use logits::{LogitsProcessor, MaskContribution, ProcCtx, ProcessorDeclaration};
pub use policy::{DecisionLog, LatencyHistory, PolicyDecision, PolicyReason, PolicySnapshot};
pub use queue::{FcfsRequestQueue, PriorityRequestQueue, RequestQueue};
pub use resources::{CreditExhausted, CreditLedger, LedgerStats};
pub use scheduler::{
    ControlTokens, DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH,
    DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS,
    HealthSnapshot, MAX_SPEC_DECODE_POS_STATS, SchedStats, Scheduler, SchedulerConfig,
    SchedulingPolicy,
};
pub use stats_report::SchedStatsReporter;
pub use trace::{RequestTrace, TraceEvent, TraceEventKind};
pub use uniserve_executor::{ControlAck, ControlOp, Executor};
