//! Transfer admission, readable results, and physical retirement.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::{Completion, Error, Outcome, Result};

struct CapacityState<C> {
    used: u64,
    reads_free: usize,
    returns: u64,
    waiters: Vec<C>,
}

/// Shared byte reservations and physical read credits for one worker rank.
/// The backend dispatches notifications after the capacity lock is released.
pub struct TransferCapacity<C> {
    capacity: u64,
    ticket_capacity: usize,
    state: Mutex<CapacityState<C>>,
    notify: fn(Vec<C>),
}

impl<C> TransferCapacity<C> {
    pub fn new(capacity: u64, ticket_capacity: usize, notify: fn(Vec<C>)) -> Result<Self> {
        if capacity == 0 || ticket_capacity == 0 {
            return Err(Error::Invalid(
                "transfer byte and read capacities must be positive".into(),
            ));
        }

        Ok(Self {
            capacity,
            ticket_capacity,
            state: Mutex::new(CapacityState {
                used: 0,
                reads_free: ticket_capacity,
                returns: 0,
                waiters: Vec::new(),
            }),
            notify,
        })
    }

    fn state(&self) -> MutexGuard<'_, CapacityState<C>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn ticket_capacity(&self) -> usize {
        self.ticket_capacity
    }

    pub fn used(&self) -> u64 {
        self.state().used
    }

    pub fn acquire(&self, amount: u64) -> Result<()> {
        let mut state = self.state();
        if amount > self.capacity - state.used {
            return Err(Error::Resource("transfer byte capacity is exhausted"));
        }

        state.used += amount;
        Ok(())
    }

    pub fn release(&self, amount: u64) -> Result<()> {
        let mut state = self.state();
        if amount > state.used {
            return Err(Error::State(
                "transfer byte release exceeds the live reservation",
            ));
        }

        state.used -= amount;
        Ok(())
    }

    /// Take the complete fan-out before starting any reads. Backpressure
    /// includes the return sequence used to subscribe without missing a wake.
    pub fn take_reads(&self, count: usize) -> Result<()> {
        if count == 0 {
            return Err(Error::Invalid(
                "a read ticket reservation takes at least one".into(),
            ));
        }
        if count > self.ticket_capacity {
            return Err(Error::Unsupported(format!(
                "a fetch of {count} reads exceeds the rank's {} read tickets",
                self.ticket_capacity
            )));
        }

        let mut state = self.state();
        if count > state.reads_free {
            return Err(Error::ReadBackpressure {
                returns: state.returns,
            });
        }

        state.reads_free -= count;
        Ok(())
    }

    /// Return credits only after a physical read retires, or if unused.
    pub fn return_reads(&self, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        let mut state = self.state();
        if count > self.ticket_capacity - state.reads_free {
            return Err(Error::State("read ticket return exceeds the tickets taken"));
        }

        state.reads_free += count;
        state.returns = state.returns.wrapping_add(1);
        let callbacks = std::mem::take(&mut state.waiters);
        drop(state);

        (self.notify)(callbacks);
        Ok(())
    }

    /// Register for a credit return, or return the callback for immediate
    /// dispatch if credits have already returned since the refused fetch.
    pub fn notify_reads_returned(&self, callback: C, after: u64) -> Option<C> {
        let mut state = self.state();
        if state.returns != after {
            return Some(callback);
        }

        state.waiters.push(callback);
        None
    }

    /// Trace retained backend callbacks. The visitor must not invoke user
    /// code or re-enter the capacity while its lock is held.
    pub fn visit<R>(&self, visitor: impl FnOnce(&[C]) -> R) -> R {
        visitor(&self.state().waiters)
    }
}

/// A fetch's reserved read credits. Each submission takes one credit; close
/// and destruction return only those which have not been handed to a read.
pub struct ReadReservation<C> {
    capacity: Arc<TransferCapacity<C>>,
    unused: AtomicUsize,
}

impl<C> ReadReservation<C> {
    pub fn new(capacity: Arc<TransferCapacity<C>>, count: usize) -> Result<Self> {
        capacity.take_reads(count)?;
        Ok(Self {
            capacity,
            unused: AtomicUsize::new(count),
        })
    }

    pub fn capacity(&self) -> &Arc<TransferCapacity<C>> {
        &self.capacity
    }

    pub fn use_read(&self) -> Result<()> {
        self.unused
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |unused| {
                unused.checked_sub(1)
            })
            .map(|_| ())
            .map_err(|_| Error::State("read reservation has no ticket left"))
    }

    pub fn close(&self) -> Result<()> {
        // Only the counter crosses threads; read storage has its own fences.
        let unused = self.unused.swap(0, Ordering::Relaxed);
        self.capacity.return_reads(unused)
    }
}

impl<C> Drop for ReadReservation<C> {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            eprintln!("Failed to return unused read credits: {error}");
        }
    }
}

/// A read's result and physical lifetime. The transport serializes mutations
/// and dispatches returned callbacks after releasing its ownership lock.
///
/// A consumable view can be followed by a device failure, so result readiness
/// is separate from the one-shot physical retirement completion.
pub struct TransferTicket<V, E, C> {
    value: Option<Arc<V>>,
    error: Option<Arc<E>>,
    cancelled: bool,
    borrowed: bool,
    closed: bool,
    undrained: bool,
    callbacks: Vec<C>,
    pub retirement: Arc<Completion<E, C>>,
}

impl<V, E, C> TransferTicket<V, E, C> {
    pub fn new(borrowed: bool) -> Self {
        Self {
            value: None,
            error: None,
            cancelled: false,
            borrowed,
            closed: false,
            undrained: false,
            callbacks: Vec::new(),
            retirement: Arc::new(Completion::default()),
        }
    }

    pub fn ready(&self) -> bool {
        self.value.is_some() || self.error.is_some()
    }

    pub fn retired(&self) -> bool {
        self.retirement.succeeded()
    }

    pub fn undrained(&self) -> bool {
        self.undrained
    }

    pub fn closed(&self) -> bool {
        self.closed
    }

    pub fn retirement_ready(&self) -> Result<bool> {
        if self.undrained {
            return Err(Error::Resource("transfer physical completion is unknown"));
        }

        Ok(self.retired())
    }

    pub fn add_done_callback(&mut self, callback: C) -> Option<C> {
        if self.ready() {
            return Some(callback);
        }

        self.callbacks.push(callback);
        None
    }

    /// Cancellation revokes consumption, but does not end a submitted access.
    pub fn cancel(&mut self, error: E) -> Vec<C> {
        self.cancelled = true;
        if self.error.is_none() {
            self.error = Some(Arc::new(error));
        }

        std::mem::take(&mut self.callbacks)
    }

    pub fn cancellation_error(&self) -> Option<Arc<E>> {
        if self.cancelled {
            self.error.clone()
        } else {
            None
        }
    }

    /// Expose a completed view without replacing an earlier error or result.
    pub fn complete(&mut self, value: V) -> Vec<C> {
        if self.ready() {
            return Vec::new();
        }

        self.value = Some(Arc::new(value));
        std::mem::take(&mut self.callbacks)
    }

    /// Return whether consumers could already have observed a successful view.
    /// Such a failure also invalidates the transport pool, not just this read.
    pub fn fail(&mut self, error: E) -> (bool, Vec<C>) {
        let late = self.value.is_some();
        self.error = Some(Arc::new(error));
        (late, std::mem::take(&mut self.callbacks))
    }

    pub fn result(&self) -> Result<Outcome<E, Arc<V>>> {
        if let Some(error) = &self.error {
            return Ok(Outcome::Failed(Arc::clone(error)));
        }

        let value = self.value.as_ref().ok_or(Error::State(
            "transfer ticket was observed before readiness",
        ))?;
        if self.closed {
            return Err(Error::State("transfer consumption has already closed"));
        }

        Ok(Outcome::Success(Arc::clone(value)))
    }

    /// End borrowed consumption once. Copy destinations retain their ordinary
    /// value ownership and remain readable after close.
    pub fn close(&mut self) -> bool {
        if !self.borrowed || self.closed {
            return false;
        }

        self.closed = true;
        true
    }

    /// The backend could not drain device access. It must retain its backing
    /// allocations; neither cancellation nor result failure permits reuse.
    pub fn mark_undrained(&mut self) {
        self.undrained = true;
    }

    pub fn retire(&self) -> Result<Vec<C>> {
        if self.undrained {
            return Err(Error::Invariant(
                "transfer cannot retire with unknown physical completion".into(),
            ));
        }

        self.retirement.complete(Ok(()))
    }

    pub fn value(&self) -> Option<&V> {
        self.value.as_deref()
    }

    pub fn error(&self) -> Option<&E> {
        self.error.as_deref()
    }

    /// Trace result references and pending readiness observers. Physical
    /// retirement observers are traced through their shared Completion.
    pub fn visit<R>(&self, visitor: impl FnOnce(Option<&V>, Option<&E>, &[C]) -> R) -> R {
        visitor(self.value(), self.error(), &self.callbacks)
    }
}
