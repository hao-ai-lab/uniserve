//! Workspace-level benchmark support for UniServe.
//!
//! Benchmark binaries and shared helpers live here when they span multiple
//! production crates.

#![deny(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub mod histograms;
pub mod native_events;
pub mod synthetic_payloads;
pub mod traces;
