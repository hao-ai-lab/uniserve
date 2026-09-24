//! Semantic terminal and stop reasons for decoded output.

pub use uniserve_core::StopReason;

/// Terminal reason for decoded text and structured chat output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinishReason {
    reason: uniserve_core::FinishReason,
    stop_reason: Option<StopReason>,
}

impl FinishReason {
    /// Constructs a terminal reason without a concrete stop cause.
    pub fn new(reason: uniserve_core::FinishReason) -> Self {
        Self {
            reason,
            stop_reason: None,
        }
    }

    /// Constructs a terminal reason with an optional concrete stop cause, the
    /// matched stop token or string.
    pub fn with_stop_reason(
        reason: uniserve_core::FinishReason,
        stop_reason: Option<StopReason>,
    ) -> Self {
        Self {
            reason,
            stop_reason,
        }
    }

    /// Constructs an end-of-sequence stop result.
    pub fn stop_eos() -> Self {
        Self::new(uniserve_core::FinishReason::Eos)
    }

    /// Returns the engine-level terminal reason.
    pub fn reason(&self) -> &uniserve_core::FinishReason {
        &self.reason
    }

    /// Borrows the concrete stop cause, if one was recorded.
    pub fn as_stop_reason(&self) -> Option<&StopReason> {
        self.stop_reason.as_ref()
    }
}
