//! Rank-local request execution, independent of the numerical backend.

mod error;
mod executor;
mod request;

pub use error::{Error, Result};
pub use executor::{Backend, Batch, Executor, Submission};
pub use request::{Request, RequestPool, RequestProgress};
