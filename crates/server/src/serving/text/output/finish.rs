/// Semantic stop cause preserved above engine transport details.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    TokenId(u32),
    Text(String),
}

/// Terminal reason for decoded text and structured chat output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinishReason {
    reason: uniserve_core::FinishReason,
    stop_reason: Option<StopReason>,
}

impl FinishReason {
    pub fn new(reason: uniserve_core::FinishReason) -> Self {
        Self {
            reason,
            stop_reason: None,
        }
    }

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

    pub fn stop_eos() -> Self {
        Self::new(uniserve_core::FinishReason::Eos)
    }

    pub fn reason(&self) -> &uniserve_core::FinishReason {
        &self.reason
    }

    pub fn as_str(&self) -> &'static str {
        match &self.reason {
            uniserve_core::FinishReason::Eos
            | uniserve_core::FinishReason::Stop
            | uniserve_core::FinishReason::ImageDone => "stop",
            uniserve_core::FinishReason::MaxTokens => "length",
            uniserve_core::FinishReason::Cancelled => "cancelled",
            uniserve_core::FinishReason::Aborted => "aborted",
            uniserve_core::FinishReason::Error => "error",
            uniserve_core::FinishReason::Repetition => "repetition",
        }
    }

    pub fn as_stop_reason(&self) -> Option<&StopReason> {
        self.stop_reason.as_ref()
    }

    pub fn into_stop_reason(self) -> Option<StopReason> {
        self.stop_reason
    }
}
