//! Rank-local request execution, independent of the numerical backend.

mod block_tables;
mod completion;
mod error;
mod executor;
mod kv_cache;
mod request;

pub use block_tables::{BlockTableUpdate, BlockTables, GroupShape, GroupTable};
pub use completion::{Completion, Outcome};
pub use error::{Error, Result};
pub use executor::{Backend, Batch, Executor, Submission};
pub use kv_cache::KVCacheManager;
pub use request::{Request, RequestPool, RequestProgress};
