//! the scheduler-stats aggregator is owned by the `scheduler` crate
//! (`uniserve_scheduler::SchedStatsReporter`) and shared verbatim by both the
//! in-process server and this headless engine process. This module is a thin
//! re-export so the mapping is defined exactly once.

pub(crate) use uniserve_scheduler::SchedStatsReporter;
