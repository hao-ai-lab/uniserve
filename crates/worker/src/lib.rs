//! Rank-local request execution, independent of the numerical backend.

mod block_tables;
mod completion;
mod error;
mod executor;
mod host;
mod kv_cache;
mod request;
mod transfer;

pub use block_tables::{BlockTableUpdate, BlockTables, GroupShape, GroupTable};
pub use completion::{Completion, Outcome};
pub use error::{Error, Result};
pub use executor::{Backend, Batch, Executor, Submission};
pub use host::{HostAction, HostLane, HostTask};
pub use kv_cache::KVCacheManager;
pub use request::{Request, RequestPool, RequestProgress};
pub use transfer::{ReadReservation, TransferCapacity, TransferTicket};
