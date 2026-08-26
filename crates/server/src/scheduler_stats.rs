//! the scheduler-stats aggregator is owned by the `scheduler` crate
//! (`uniserve_engine::scheduler::SchedStatsReporter`) and shared verbatim by both this
//! in-process server and the headless engine process. This module is a thin
//! re-export so the mapping is defined exactly once.

pub(crate) use uniserve_engine::scheduler::SchedStatsReporter;
