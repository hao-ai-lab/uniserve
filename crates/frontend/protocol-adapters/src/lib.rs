//! Protocol edge adapters over the semantic serving runtime.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod grpc;
pub mod native;
pub mod openai;
pub mod raw_generate;

pub mod runtime {
    pub use uniserve_serving::{
        AdapterSelection, CachePolicy, ContextSegment, ExecutionPlan, GenerationPolicy,
        ModelContext, ServeEvent, ServeRequest, ServeRequestId, ServingRuntime,
    };
}
