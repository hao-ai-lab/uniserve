/// Semantic stop cause preserved above engine transport details.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    TokenId(u32),
    Text(String),
}

/// Terminal reason for decoded text and structured chat output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop(Option<StopReason>),
    Length,
    Abort,
    Cancelled,
    Aborted,
    Error,
    Repetition,
}

impl FinishReason {
    pub fn stop_eos() -> Self {
        Self::Stop(None)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stop(_) => "stop",
            Self::Length => "length",
            Self::Abort => "abort",
            Self::Cancelled => "cancelled",
            Self::Aborted => "aborted",
            Self::Error => "error",
            Self::Repetition => "repetition",
        }
    }

    pub fn as_stop_reason(&self) -> Option<&StopReason> {
        match self {
            Self::Stop(reason) => reason.as_ref(),
            _ => None,
        }
    }

    pub fn into_stop_reason(self) -> Option<StopReason> {
        match self {
            Self::Stop(reason) => reason,
            _ => None,
        }
    }
}
