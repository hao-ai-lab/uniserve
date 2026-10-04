//! Rank-local request execution, independent of the numerical backend.

mod error;
mod request;

pub use error::{Error, Result};
pub use request::{Request, RequestPool, RequestProgress};
