//! Transport-level timestamp helpers.

/// Returns the current Unix timestamp in whole seconds for API response objects.
pub(crate) fn unix_timestamp() -> u64 {
    uniserve_core::now_unix_secs_u64()
}
