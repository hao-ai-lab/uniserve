//! Rank-local request execution, independent of the numerical backend.

mod block_tables;
mod buffer;
mod completion;
pub mod cuda;
mod descriptor_grants;
mod error;
mod events;
mod executor;
mod expert_exchange;
mod graph_storage;
mod host;
mod host_buffers;
mod inputs;
mod kv_cache;
mod kv_import;
mod latent;
mod microbatches;
mod output;
mod pending;
mod profiling;
mod registry;
mod request;
mod service;
mod shared_buffer;
mod stream;
pub mod tensor;
mod transfer;
pub mod vmm_pool;
mod weight_prefetch;

pub use block_tables::{BlockTableUpdate, BlockTables, GroupShape, GroupTable};
pub use buffer::{BufferBinding, BufferPool};
pub use completion::{Completion, Outcome};
pub use descriptor_grants::{DescriptorGrants, fetch_descriptor};
pub use error::{Error, Result};
pub use events::EventPool;
pub use executor::{Backend, Batch, Executor, Submission};
pub use expert_exchange::ExpertExchange;
pub use graph_storage::{GraphPool, GraphPools, GraphStorage, graph_storage_budget_bytes};
pub use host::{HostAction, HostLane, HostTask};
pub use host_buffers::HostBuffers;
pub use inputs::{BatchInputs, InputReady, InputWait};
pub use kv_cache::KVCacheManager;
pub use kv_import::{KVImport, KVImporter};
pub use latent::{LatentExport, LatentImport, LatentPool, LatentUpdate};
pub use microbatches::{Microbatches, yield_microbatch};
pub use output::{LogprobLayout, OutputBuffer, OutputPool, OutputStorage};
pub use pending::{BatchResult, PendingOutput, request_output};
pub use registry::{BufferRegistry, RegisteredBuffer};
pub use request::{Request, RequestPool, RequestProgress};
pub use service::{Service, ServiceBackend};
pub use shared_buffer::{SHM_HEADER_BYTES, SharedBuffer, SharedMapping, SharedRead};
pub use stream::CUDAStream;
pub use transfer::{
    ReadRegion, ReadReservation, TransferCapacity, TransferPool, TransferTicket, plan_reads,
};
pub use weight_prefetch::WeightPrefetch;
