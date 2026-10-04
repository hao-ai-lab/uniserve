//! Emit a request-retirement submission for the language-boundary test.

use std::io::Write;

use uniserve_core::{ImageParams, RequestId};
use uniserve_worker_ipc::{codec, Batch, BatchCommand, NewRequest, RequestKey, WorkerRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = RequestKey::new(1, RequestId(5), 1);
    let admission = NewRequest::new(key, 2, None, Some(ImageParams::default()), 0)?;
    let mut batch = Batch::new(17, vec![admission], Vec::new());
    batch.commands.push(BatchCommand::Finish {
        request_key: key,
        retained_buffers: Vec::new(),
    });
    let request = WorkerRequest::Submit {
        message_id: Some(23),
        batch: Box::new(batch),
    };
    std::io::stdout().write_all(&codec::encode_request(&request)?)?;
    Ok(())
}
