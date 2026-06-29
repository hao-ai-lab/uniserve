//! Per-request output stream channel types shared by the client backends and
//! [`super::stream::EngineCoreOutputStream`].

use tokio::sync::{mpsc, oneshot};

use crate::Result;
use crate::client::stream::EngineCoreStreamOutput;
use crate::protocol::utility::UtilityOutput;

/// Sender half of one request's output stream channel.
pub(crate) type OutputSender = mpsc::UnboundedSender<Result<EngineCoreStreamOutput>>;
/// Receiver half of one request's output stream channel.
pub(crate) type OutputReceiver = mpsc::UnboundedReceiver<Result<EngineCoreStreamOutput>>;
/// Sender half of one utility call's result channel.
pub(crate) type UtilitySender = oneshot::Sender<Result<UtilityOutput>>;
/// Receiver half of one utility call's result channel.
pub(crate) type UtilityReceiver = oneshot::Receiver<Result<UtilityOutput>>;
