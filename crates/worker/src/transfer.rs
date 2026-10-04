//! Rank-wide storage and read budgets shared by transport backends.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::{Error, Result};

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
