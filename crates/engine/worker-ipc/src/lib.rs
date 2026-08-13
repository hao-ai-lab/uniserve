//! Worker IPC implementations over iceoryx2 request-response services.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod death_watch;
mod multiproc;
mod stage_router;
mod uniproc;

pub use multiproc::MultiprocExecutor;
pub use stage_router::StageRouter;
pub use uniproc::{LaneConfig, UniprocExecutor, WorkerLaunchConfig};
