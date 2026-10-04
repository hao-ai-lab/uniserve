//! Resources retained by a batch while its numerical inputs are prepared.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use uniserve_worker_ipc::{BufferId, CallId};

/// A borrowed resource reports consumability; physical retirement remains
/// the responsibility of its storage owner. Queries neither submit work nor
/// invoke completion observers while the caller is borrowing its input set.
pub trait InputReady {
    type Error;

    fn ready(&self) -> Result<bool, Self::Error>;
}

/// Input leases and preparation progress for one scheduler batch. Backends
/// retain their numerical views in I and their completion owners in D.
pub struct BatchInputs<I, D> {
    pub inputs: HashMap<BufferId, I>,
    pub tasks: HashMap<CallId, D>,
    pub dependencies: Vec<D>,
    pub predicate: Option<D>,
    pub started: usize,
    pub submitted: bool,
    pub awaiting_reads: bool,
    closed: Arc<AtomicBool>,
}

impl<I, D> Default for BatchInputs<I, D> {
    fn default() -> Self {
        Self {
            inputs: HashMap::new(),
            tasks: HashMap::new(),
            dependencies: Vec::new(),
            predicate: None,
            started: 0,
            submitted: false,
            awaiting_reads: false,
            closed: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl<I, D> BatchInputs<I, D> {
    pub fn closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Claim cleanup before releasing any resource. A failed release does
    /// not prevent the remaining owners from being closed by the backend.
    pub fn close(&self) -> bool {
        !self.closed.swap(true, Ordering::AcqRel)
    }

    /// The extra arrival prevents an already-complete dependency from firing
    /// the observer before registration has finished.
    pub fn wait(&self, dependencies: usize) -> Arc<InputWait> {
        Arc::new(InputWait {
            remaining: AtomicUsize::new(dependencies + 1),
            closed: Arc::clone(&self.closed),
        })
    }
}

impl<I, D> BatchInputs<I, D>
where
    I: InputReady,
    D: InputReady<Error = I::Error>,
{
    pub fn storage_ready(&self) -> Result<bool, I::Error> {
        for dependency in &self.dependencies {
            if !dependency.ready()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn ready(&self) -> Result<bool, I::Error> {
        if !self.submitted || !self.storage_ready()? {
            return Ok(false);
        }
        for task in self.tasks.values() {
            if !task.ready()? {
                return Ok(false);
            }
        }
        for input in self.inputs.values() {
            if !input.ready()? {
                return Ok(false);
            }
        }
        if let Some(predicate) = &self.predicate {
            return predicate.ready();
        }
        Ok(true)
    }

    pub fn input_ready(&self, buffer: &BufferId) -> Result<bool, I::Error> {
        self.inputs.get(buffer).map_or(Ok(false), InputReady::ready)
    }
}

/// One snapshot of input dependencies. Each dependency arrives exactly once;
/// only the last arrival may notify the executor, and closing suppresses it.
pub struct InputWait {
    remaining: AtomicUsize,
    closed: Arc<AtomicBool>,
}

impl InputWait {
    pub fn arrive(&self) -> bool {
        self.remaining.fetch_sub(1, Ordering::AcqRel) == 1 && !self.closed.load(Ordering::Acquire)
    }
}
