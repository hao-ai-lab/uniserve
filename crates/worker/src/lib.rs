//! Rank-local request execution, independent of the numerical backend.

mod block_tables;
mod error;
mod executor;
mod request;

pub use block_tables::{BlockTableUpdate, BlockTables, GroupShape, GroupTable};
pub use error::{Error, Result};
pub use executor::{Backend, Batch, Executor, Submission};
pub use request::{Request, RequestPool, RequestProgress};
