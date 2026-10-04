//! Shared operation results, waits, and completion observers.

use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use crate::{Error, Result};

/// The operation's result. Physical storage requires a successful retirement
/// signal; a task result alone does not establish that a device has stopped.
#[derive(Debug)]
pub enum Outcome<E, T = ()> {
    Success(T),
    Failed(Arc<E>),
    Cancelled,
}

impl<E, T: Clone> Clone for Outcome<E, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Success(value) => Self::Success(value.clone()),
            Self::Failed(error) => Self::Failed(Arc::clone(error)),
            Self::Cancelled => Self::Cancelled,
        }
    }
}

struct State<E, C, T> {
    outcome: Option<Outcome<E, T>>,
    callbacks: Vec<C>,
}

/// One completion shared by an operation and its consumers.
///
/// Errors and callbacks belong to the backend. Completing an operation returns
/// its callbacks so the backend can dispatch them after releasing the lock.
pub struct Completion<E, C, T = ()> {
    state: Mutex<State<E, C, T>>,
    ready: Condvar,
}

impl<E, C, T> Default for Completion<E, C, T> {
    fn default() -> Self {
        Self {
            state: Mutex::new(State {
                outcome: None,
                callbacks: Vec::new(),
            }),
            ready: Condvar::new(),
        }
    }
}

impl<E, C, T: Clone> Completion<E, C, T> {
    fn state(&self) -> MutexGuard<'_, State<E, C, T>> {
        // No user code mutates state under the lock. A panicking reference
        // visitor cannot leave a partially completed operation.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn done(&self) -> bool {
        self.state().outcome.is_some()
    }

    pub fn succeeded(&self) -> bool {
        matches!(self.state().outcome, Some(Outcome::Success(_)))
    }

    pub fn outcome(&self) -> Option<Outcome<E, T>> {
        self.state().outcome.clone()
    }

    /// Return None only if the timeout expires before completion.
    pub fn wait(&self, timeout: Option<Duration>) -> Option<Outcome<E, T>> {
        let state = self.state();
        let pending = |state: &mut State<E, C, T>| state.outcome.is_none();
        let state = match timeout {
            Some(timeout) => {
                self.ready
                    .wait_timeout_while(state, timeout, pending)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0
            }
            None => self
                .ready
                .wait_while(state, pending)
                .unwrap_or_else(PoisonError::into_inner),
        };
        state.outcome.clone()
    }

    /// Register a callback, or return it for immediate dispatch if already done.
    pub fn subscribe(&self, callback: C) -> Option<C> {
        let mut state = self.state();
        if state.outcome.is_some() {
            return Some(callback);
        }

        state.callbacks.push(callback);
        None
    }

    pub fn complete(&self, result: std::result::Result<T, E>) -> Result<Vec<C>> {
        let mut state = self.state();
        if state.outcome.is_some() {
            return Err(Error::State("operation has already completed"));
        }

        state.outcome = Some(match result {
            Ok(value) => Outcome::Success(value),
            Err(error) => Outcome::Failed(Arc::new(error)),
        });
        let callbacks = std::mem::take(&mut state.callbacks);
        self.ready.notify_all();
        Ok(callbacks)
    }

    /// Cancel an unresolved completion. This does not stop its producer.
    pub fn cancel(&self) -> (bool, Vec<C>) {
        let mut state = self.state();
        match state.outcome {
            Some(Outcome::Cancelled) => return (true, Vec::new()),
            Some(_) => return (false, Vec::new()),
            None => {}
        }

        state.outcome = Some(Outcome::Cancelled);
        let callbacks = std::mem::take(&mut state.callbacks);
        self.ready.notify_all();
        (true, callbacks)
    }

    /// Trace backend references for garbage collection. The visitor must not
    /// invoke user code or re-enter this completion while the lock is held.
    pub fn visit<R>(&self, visitor: impl FnOnce(Option<&Outcome<E, T>>, &[C]) -> R) -> R {
        let state = self.state();
        visitor(state.outcome.as_ref(), &state.callbacks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::thread;

    #[test]
    fn waiters_and_concurrent_subscribers_observe_completion()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let completion = Completion::<String, mpsc::Sender<()>>::default();
        let (tx, rx) = mpsc::channel();

        thread::scope(
            |scope| -> std::result::Result<(), Box<dyn std::error::Error>> {
                let waiter = scope.spawn(|| completion.wait(None));
                let subscriber = scope.spawn(|| {
                    if let Some(callback) = completion.subscribe(tx) {
                        callback.send(())?;
                    }
                    Ok::<(), mpsc::SendError<()>>(())
                });

                for callback in completion.complete(Ok(()))? {
                    callback.send(())?;
                }
                rx.recv()?;
                assert!(matches!(
                    waiter.join().map_err(|_| "waiter panicked")?,
                    Some(Outcome::Success(()))
                ));
                subscriber.join().map_err(|_| "subscriber panicked")??;
                Ok(())
            },
        )?;

        assert!(completion.succeeded());
        assert!(!completion.cancel().0);
        assert!(completion.complete(Ok(())).is_err());
        Ok(())
    }

    #[test]
    fn timeout_failure_and_cancellation_do_not_retire_storage() -> Result<()> {
        let failed = Completion::<String, ()>::default();
        assert!(failed.wait(Some(Duration::ZERO)).is_none());
        assert!(!failed.done());

        failed.complete(Err("device completion unknown".into()))?;
        assert!(failed.done());
        assert!(!failed.succeeded());
        let Some(Outcome::Failed(error)) = failed.wait(None) else {
            panic!("lost the producer error");
        };
        assert_eq!(*error, "device completion unknown");

        let cancelled = Completion::<String, ()>::default();
        assert!(cancelled.cancel().0);
        assert!(cancelled.done());
        assert!(!cancelled.succeeded());
        assert!(matches!(cancelled.wait(None), Some(Outcome::Cancelled)));
        assert!(cancelled.cancel().0);
        assert!(cancelled.complete(Ok(())).is_err());
        Ok(())
    }
}
