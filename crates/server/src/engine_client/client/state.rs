//! Per-request canonical event channels shared by client backends.

use tokio::sync::mpsc;

use uniserve_core::GenEvent;

pub(crate) type EventSender = mpsc::Sender<GenEvent>;
pub(crate) type EventReceiver = mpsc::Receiver<GenEvent>;
