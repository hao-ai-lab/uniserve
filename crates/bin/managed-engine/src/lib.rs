//! Supervision of headless `uniserve engine` subprocesses (managed mode).
//!
//! The supervisor spawns each engine in its own process group (so teardown
//! reaps the engine's Python worker too), polls for unexpected exits, and
//! shuts down with SIGTERM → bounded wait → SIGKILL.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod process;

pub use process::{ManagedEngineConfig, ManagedEngineHandle, allocate_handshake_port};
