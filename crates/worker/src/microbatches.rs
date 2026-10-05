//! Cooperative host turns for numerical microbatches.

use std::cell::RefCell;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use crate::{Error, Result};

thread_local! {
    static CURRENT: RefCell<Option<(Arc<Microbatches>, usize)>> = const { RefCell::new(None) };
}

struct State {
    active: Vec<bool>,
    turn: usize,
    running: bool,
    closed: bool,
    aborted: bool,
    failure: Option<usize>,
}

/// Rotate ordinary numerical calls at expert dispatch points.
///
/// Each index runs on its own persistent host lane. Calls start in index order,
/// yield to the next active index and leave the rotation when they return.
/// The caller joins all host tasks before ending an invocation or closing.
pub struct Microbatches {
    state: Mutex<State>,
    ready: Condvar,
}

impl Microbatches {
    pub fn new(count: usize) -> Result<Self> {
        if count == 0 {
            return Err(Error::Invalid(
                "microbatches require at least one context".into(),
            ));
        }

        Ok(Self {
            state: Mutex::new(State {
                active: vec![false; count],
                turn: 0,
                running: false,
                closed: false,
                aborted: false,
                failure: None,
            }),
            ready: Condvar::new(),
        })
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn begin(&self, count: usize) -> Result<()> {
        let mut state = self.state();
        if count != state.active.len() {
            return Err(Error::Invalid(
                "each microbatch context needs one numerical call".into(),
            ));
        }
        if state.running {
            return Err(Error::State("microbatch invocation is already running"));
        }
        if state.closed {
            return Err(Error::State("microbatch execution is closed"));
        }

        state.active.fill(true);
        state.turn = 0;
        state.running = true;
        state.aborted = false;
        state.failure = None;
        Ok(())
    }

    /// Execute one admitted index, once per invocation. The caller starts
    /// every index before joining any of them. Waiting and yielding require
    /// no language runtime lock; only the numerical callback acquires it.
    pub fn run<T, E>(
        self: &Arc<Self>,
        index: usize,
        call: impl FnOnce() -> std::result::Result<T, E>,
    ) -> Result<std::result::Result<T, E>> {
        let previous = CURRENT.replace(Some((Arc::clone(self), index)));
        let mut turn = Turn {
            owner: Arc::clone(self),
            index,
            previous,
            failed: true,
        };
        self.wait(index, self.state())?;
        let result = call();
        turn.failed = result.is_err();
        Ok(result)
    }

    fn wait(&self, index: usize, state: MutexGuard<'_, State>) -> Result<()> {
        let state = self
            .ready
            .wait_while(state, |state| state.turn != index && !state.aborted)
            .unwrap_or_else(PoisonError::into_inner);
        if state.aborted {
            return Err(Error::State("microbatch execution aborted"));
        }
        Ok(())
    }

    fn resume(&self, index: usize) -> Result<()> {
        let mut state = self.state();
        next(&mut state, index);
        self.ready.notify_all();
        self.wait(index, state)
    }

    /// Interrupt suspended calls. True means this is the first failure;
    /// otherwise the failing task remains the source of the invocation error.
    pub fn abort(&self) -> bool {
        let mut state = self.state();
        let first = !state.aborted;
        state.aborted = true;
        self.ready.notify_all();
        first
    }

    pub fn failure(&self) -> Option<usize> {
        self.state().failure
    }

    /// Complete an invocation after joining its host turns and device streams.
    pub fn end(&self) {
        let mut state = self.state();
        state.running = false;
        state.active.fill(false);
        state.failure = None;
    }

    pub fn close(&self) -> Result<()> {
        let mut state = self.state();
        if state.running {
            return Err(Error::State("cannot close running microbatches"));
        }
        state.closed = true;
        Ok(())
    }
}

/// Yield after submitting an expert dispatch. Outside a microbatch call this
/// is a no-op, so numerical layers use the same operation in ordinary forwards.
pub fn yield_microbatch() -> Result<()> {
    let current = CURRENT.with_borrow(Clone::clone);
    if let Some((owner, index)) = current {
        owner.resume(index)?;
    }
    Ok(())
}

fn next(state: &mut State, index: usize) {
    let count = state.active.len();
    if let Some(next) = (1..=count)
        .map(|offset| (index + offset) % count)
        .find(|&next| state.active[next])
    {
        state.turn = next;
    }
}

/// Retire a turn on ordinary return, callback error or unwinding. Restoring
/// the thread-local owner also permits nested independent numerical invocations.
struct Turn {
    owner: Arc<Microbatches>,
    index: usize,
    previous: Option<(Arc<Microbatches>, usize)>,
    failed: bool,
}

impl Drop for Turn {
    fn drop(&mut self) {
        CURRENT.set(self.previous.take());
        let mut state = self.owner.state();
        if self.failed && !state.aborted {
            state.failure = Some(self.index);
            state.aborted = true;
        }
        state.active[self.index] = false;
        next(&mut state, self.index);
        self.owner.ready.notify_all();
    }
}
