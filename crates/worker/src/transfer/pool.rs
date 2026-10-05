//! Read tasks, physical credit returns and per-thread copy streams.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError};
use std::thread::ThreadId;

use crate::cuda::{DeviceGuard, Stream};
use crate::{Error, HostAction, HostLane, HostTask, Outcome, Result};

use super::{ReadReservation, TransferCapacity, TransferTicket};

/// Numerical read execution and observers supplied by the language binding.
/// `read` returns after physical access ends. If completion cannot be established,
/// it marks its ticket undrained and returns an error.
/// Ticket access is serialized without invoking numerical code or observers.
pub trait ReadBackend: Send + Sync + 'static {
    type Value: Send + Sync;
    type Error: Send + Sync;
    type Callback: Send + Sync;

    fn with_ticket<T>(
        &self,
        action: impl FnOnce(&mut TransferTicket<Self::Value, Self::Error, Self::Callback>) -> T,
    ) -> T;

    fn read(&self) -> std::result::Result<(), Self::Error>;
    fn notify(callbacks: Vec<Self::Callback>);
    fn wake(callback: &Self::Callback);
    fn error(error: Error) -> Self::Error;
    fn report(error: Self::Error);
}

/// A read action submitted through the common native host executor.
pub struct TransferRead<B: ReadBackend> {
    backend: Arc<B>,
}

impl<B: ReadBackend> HostAction for TransferRead<B> {
    type Output = ();
    type Error = B::Error;
    type Callback = Box<dyn FnOnce() + Send>;
    type Wake = ();

    fn ready(&self) -> std::result::Result<bool, Self::Error> {
        Ok(true)
    }

    fn run(&self) -> std::result::Result<(), Self::Error> {
        if self.backend.with_ticket(|ticket| ticket.is_cancelled()) {
            return Err(B::error(Error::Resource("transfer read was cancelled")));
        }

        let _range = crate::profiling::range(c"uniserve.transfer", None);
        self.backend.read()
    }

    fn release(&self) -> std::result::Result<(), Self::Error> {
        Ok(())
    }

    fn input_outcome(&self) -> Option<Outcome<Self::Error>> {
        // Backend reads own their input accesses through read completion.
        Some(Outcome::Success(()))
    }

    fn defer_release(_: Arc<HostTask<Self>>) -> std::result::Result<(), Self::Error> {
        Err(B::error(Error::Invariant(
            "transfer inputs retire with their read task".into(),
        )))
    }

    fn notify(callbacks: Vec<Self::Callback>) {
        for callback in callbacks {
            callback();
        }
    }

    fn wake(_: &()) {}

    fn report(error: Self::Error) {
        B::report(error);
    }

    fn error(error: Error) -> Self::Error {
        B::error(error)
    }

    fn note_cleanup(_: &mut Self::Error, cleanup: Self::Error) {
        B::report(cleanup);
    }
}

struct PoolState<B: ReadBackend> {
    streams: HashMap<(ThreadId, i32), Arc<Stream>>,
    error: Option<Arc<B::Error>>,
    unretired: Vec<Arc<B>>,
    wake: Option<Arc<B::Callback>>,
}

/// Execute reads within the rank's shared byte and read budgets. Cancellation
/// withdraws queued tasks; running reads retain credits until physical access
/// ends. Late failures poison admission and unknown completion retains backing.
pub struct TransferPool<B: ReadBackend> {
    capacity: Arc<TransferCapacity<B::Callback>>,
    lane: HostLane<TransferRead<B>>,
    state: Mutex<PoolState<B>>,
}

impl<B: ReadBackend> TransferPool<B> {
    pub fn new(
        capacity: Arc<TransferCapacity<B::Callback>>,
        workers: usize,
        name: &str,
    ) -> Result<Self> {
        let limit = capacity.ticket_capacity();
        Ok(Self {
            capacity,
            lane: HostLane::new(limit, workers.min(limit), name)?,
            state: Mutex::new(PoolState {
                streams: HashMap::new(),
                error: None,
                unretired: Vec::new(),
                wake: None,
            }),
        })
    }

    fn state(&self) -> MutexGuard<'_, PoolState<B>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn set_completion_wake(&self, wake: Option<B::Callback>) {
        let previous = std::mem::replace(&mut self.state().wake, wake.map(Arc::new));
        drop(previous);
    }

    pub fn completion_wake(&self) -> Option<Arc<B::Callback>> {
        self.state().wake.clone()
    }

    /// Reserve the read's credits and host task before numerical preparation.
    /// The caller then records destination handoff and submits the task, or
    /// abandons it after preparation fails. Cancellation returns unused credits.
    pub fn reserve(
        self: &Arc<Self>,
        backend: B,
        nbytes: u64,
        reservation: Option<&ReadReservation<B::Callback>>,
    ) -> Result<Arc<HostTask<TransferRead<B>>>> {
        if self.error().is_some() {
            return Err(Error::Resource("transfer pool has failed"));
        }
        if let Some(reservation) = reservation {
            if !Arc::ptr_eq(reservation.capacity(), &self.capacity) {
                return Err(Error::Invalid(
                    "read reservation belongs to another transfer capacity".into(),
                ));
            }
            reservation.use_read()?;
        } else {
            self.capacity.take_reads(1)?;
        }

        if let Err(error) = self.capacity.acquire(nbytes) {
            self.capacity.return_reads(1)?;
            return Err(error);
        }

        let backend = Arc::new(backend);
        let task = match self.lane.reserve() {
            Ok(task) => task,
            Err(error) => {
                self.return_capacity(nbytes)?;
                return Err(error);
            }
        };
        if let Err(error) = task.configure(TransferRead {
            backend: Arc::clone(&backend),
        }) {
            if let Err(cleanup) = task.cancel(true) {
                B::report(cleanup);
            }
            self.return_capacity(nbytes)?;
            return Err(error);
        }

        let observed = Arc::downgrade(&task);
        let owner = Arc::clone(self);
        let immediate = task.completion.subscribe(Box::new(move || {
            // The worker or cancelling caller retains the task while notifying.
            if let Some(task) = observed.upgrade() {
                Self::finished(&owner.state, &owner.capacity, backend, nbytes, &task);
            }
        }));
        if let Some(callback) = immediate {
            callback();
        }

        Ok(task)
    }

    fn return_capacity(&self, nbytes: u64) -> Result<()> {
        self.capacity.release(nbytes)?;
        self.capacity.return_reads(1)
    }

    fn finished(
        state: &Mutex<PoolState<B>>,
        capacity: &TransferCapacity<B::Callback>,
        backend: Arc<B>,
        nbytes: u64,
        task: &HostTask<TransferRead<B>>,
    ) {
        let error = match task.completion.outcome() {
            Some(Outcome::Success(_)) => None,
            Some(Outcome::Failed(error)) => Some(error),
            Some(Outcome::Cancelled) => Some(Arc::new(B::error(Error::Resource(
                "transfer read was cancelled before submission",
            )))),
            None => return,
        };
        let (undrained, late, callbacks) = backend.with_ticket(|ticket| {
            let (late, callbacks) = error.as_ref().map_or((false, Vec::new()), |error| {
                ticket.fail_shared(Arc::clone(error))
            });
            (ticket.undrained(), late, callbacks)
        });
        let wake = {
            let mut state = state.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(error) = error.as_ref().filter(|_| late || undrained) {
                if state.error.is_none() {
                    state.error = Some(Arc::clone(error));
                }
                if undrained {
                    state.unretired.push(Arc::clone(&backend));
                }
            }
            late.then(|| state.wake.clone()).flatten()
        };
        B::notify(callbacks);
        if let Some(wake) = wake {
            B::wake(&wake);
        }

        if !undrained {
            let retired = capacity
                .release(nbytes)
                .and_then(|()| capacity.return_reads(1))
                .and_then(|()| backend.with_ticket(|ticket| ticket.retire()));
            match retired {
                Ok(callbacks) => B::notify(callbacks),
                Err(error) => B::report(B::error(error)),
            }
        }
    }

    pub fn stream(&self, device: i32) -> Result<Arc<Stream>> {
        let key = (std::thread::current().id(), device);
        let mut state = self.state();
        if let Some(stream) = state.streams.get(&key) {
            return Ok(Arc::clone(stream));
        }

        let _device = DeviceGuard::new(device).map_err(Error::Cuda)?;
        let stream = Arc::new(Stream::new().map_err(Error::Cuda)?);
        state.streams.insert(key, Arc::clone(&stream));
        Ok(stream)
    }

    pub fn error(&self) -> Option<Arc<B::Error>> {
        self.state().error.clone()
    }

    /// Join reads before destroying their streams. Failed physical retirement
    /// remains retained and its original error is returned to the caller.
    pub fn close(&self) -> Option<Arc<B::Error>> {
        let mut errors = self.lane.close().into_iter();
        let first = errors.next().map(Arc::new);
        let error = {
            let mut state = self.state();
            if state.error.is_none() {
                state.error = first.clone();
            }
            state.streams.clear();
            state.error.clone()
        };
        for cleanup in errors {
            B::report(cleanup);
        }
        error
    }

    /// Trace only idle pool-owned references during language-runtime GC.
    pub fn visit<T>(
        &self,
        visitor: impl FnOnce(Option<&B::Error>, Option<&B::Callback>, &[Arc<B>]) -> T,
    ) -> Option<T> {
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => return None,
        };
        Some(visitor(
            state.error.as_deref(),
            state.wake.as_deref(),
            &state.unretired,
        ))
    }
}

impl<B: ReadBackend> Drop for TransferPool<B> {
    #[expect(
        clippy::mem_forget,
        reason = "unknown device completion retains backing until process teardown"
    )]
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        for read in state.unretired.drain(..) {
            std::mem::forget(read);
        }
    }
}
