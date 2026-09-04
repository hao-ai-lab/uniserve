//! Semantic terminal and stop reasons for decoded output.

/// Semantic stop cause preserved above engine transport details.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Generation matched a stop-token identifier.
    TokenId(u32),
    /// Decoded output matched a stop string.
    Text(String),
}

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

    /// Attaches the token or string that caused termination.
    pub fn with_stop_reason(
        reason: uniserve_core::FinishReason,
        stop_reason: Option<StopReason>,
    ) -> Self {
        debug_assert_eq!(reason, uniserve_core::FinishReason::Stop);
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

    /// Returns the OpenAI-compatible finish-reason string.
    pub fn as_str(&self) -> &'static str {
        match &self.reason {
            uniserve_core::FinishReason::Eos
            | uniserve_core::FinishReason::Completed
            | uniserve_core::FinishReason::Stop
            | uniserve_core::FinishReason::ImageDone => "stop",
            uniserve_core::FinishReason::MaxTokens => "length",
            uniserve_core::FinishReason::Cancelled => "cancelled",
            uniserve_core::FinishReason::Aborted => "aborted",
            uniserve_core::FinishReason::Error => "error",
            uniserve_core::FinishReason::Repetition => "repetition",
        }
    }

    /// Borrows the concrete stop cause, if one was recorded.
    pub fn as_stop_reason(&self) -> Option<&StopReason> {
        self.stop_reason.as_ref()
    }

    /// Consumes the value and returns its concrete stop cause.
    pub fn into_stop_reason(self) -> Option<StopReason> {
        self.stop_reason
    }
}
