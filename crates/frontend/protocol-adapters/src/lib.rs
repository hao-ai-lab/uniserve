//! Protocol edge adapters over the semantic serving runtime.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod native {
    pub use uniserve_native_api::*;
}

pub mod openai {
    pub use uniserve_openai_api::*;
}

pub mod runtime {
    pub use uniserve_serving::{
        AdapterSelection, CachePolicy, ExecutionPlan, GenerationPolicy, ModelContext, ServeEvent,
        ServeRequest, ServeRequestId, ServingRuntime,
    };
}
