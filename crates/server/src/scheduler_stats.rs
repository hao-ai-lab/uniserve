//! Scheduler-stats aggregator owned by `uniserve_engine::SchedStatsReporter`.
//! This module re-exports that reporter so the HTTP process uses the same mapping.

pub(crate) use uniserve_engine::SchedStatsReporter;
