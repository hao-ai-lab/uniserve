//! Worker IPC implementations over iceoryx2 request-response services.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod death_watch;
mod multiproc;
mod stage_router;
mod uniproc;

pub use multiproc::MultiprocExecutor;
// `StageRouter` is the N-pool, OpKind-routed executor; [`StageRouter::two_role`]
// is the Understanding/Generation 2-pool constructor. `TensorMover` is the
// data-plane Tier 1.
pub use stage_router::{StageRouter, TensorMover};
pub use uniproc::UniprocExecutor;
