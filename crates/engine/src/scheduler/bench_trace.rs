//! Optional JSON Lines trace sink for scheduler events.
//!
//! Set `UNISERVE_SCHED_TRACE_DIR=/path/to/dir` before starting the server to
//! write records. The scheduler owner thread writes synchronously to preserve
//! event order.

use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use serde::Serialize;

const TRACE_DIR_ENV: &str = "UNISERVE_SCHED_TRACE_DIR";

/// Lazily enabled JSON Lines writer owned by the engine thread.
pub(crate) struct RuntimeTraceSink {
    writer: BufWriter<File>,
}

impl RuntimeTraceSink {
    /// Opens the configured scheduler trace file, returning `None` when tracing is disabled or unavailable.
    pub(crate) fn from_env() -> Option<Self> {
        let dir = std::env::var_os(TRACE_DIR_ENV).map(PathBuf::from)?;
        if let Err(error) = create_dir_all(&dir) {
            tracing::warn!(path = %dir.display(), %error, "failed to create scheduler trace dir");
            return None;
        }
        let path = dir.join("scheduler_trace.jsonl");
        let file = match OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
        {
            Ok(file) => file,
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "failed to open scheduler trace file");
                return None;
            }
        };
        Some(Self {
            writer: BufWriter::new(file),
        })
    }

    /// Appends one serialized record and flushes it as a complete JSON Lines entry.
    pub(crate) fn record<T: Serialize>(&mut self, record: &T) {
        if let Err(error) = serde_json::to_writer(&mut self.writer, record) {
            tracing::warn!(%error, "failed to write scheduler trace record");
            return;
        }
        if let Err(error) = self.writer.write_all(b"\n") {
            tracing::warn!(%error, "failed to write scheduler trace newline");
        }
    }
}

impl Drop for RuntimeTraceSink {
    /// Releases resources owned by this value.
    fn drop(&mut self) {
        if let Err(error) = self.writer.flush() {
            tracing::warn!(%error, "failed to flush scheduler trace");
        }
    }
}
