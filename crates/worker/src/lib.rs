//! Rank-local request execution, independent of the numerical backend.

mod block_tables;
mod buffer;
mod completion;
pub mod cuda;
mod error;
mod events;
mod executor;
mod host;
mod kv_cache;
mod latent;
mod output;
mod registry;
mod request;
mod service;
pub mod tensor;
mod transfer;

pub use block_tables::{BlockTableUpdate, BlockTables, GroupShape, GroupTable};
pub use buffer::{BufferBinding, BufferPool};
pub use completion::{Completion, Outcome};
pub use error::{Error, Result};
pub use events::EventPool;
pub use executor::{Backend, Batch, Executor, Submission};
pub use host::{HostAction, HostLane, HostTask};
pub use kv_cache::KVCacheManager;
pub use latent::{LatentExport, LatentImport, LatentPool, LatentUpdate};
pub use output::{OutputBuffer, OutputPool, OutputStorage};
pub use registry::{BufferRegistry, RegisteredBuffer};
pub use request::{Request, RequestPool, RequestProgress};
pub use service::{Service, ServiceBackend};
pub use transfer::{ReadReservation, TransferCapacity, TransferPool, TransferTicket};
