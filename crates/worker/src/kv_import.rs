//! KV import destinations and conversion workspaces retained through physical reads.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::Deref;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, TryLockError};

use uniserve_worker_ipc::{BufferId, RequestKey};

use crate::{Completion, Error, Result};

struct ImportState<T, W> {
    reads: Vec<Arc<T>>,
    workspace: Option<Arc<W>>,
    started: bool,
    finished: bool,
    drained: bool,
    cancelled: bool,
    released: bool,
}

/// One destination held until adoption or abandonment and all physical access
/// finishes. T borrows a transfer's native retirement; W owns numerical scratch.
pub struct KVImport<T, W> {
    pub buffer: BufferId,
    pub request_pool_idx: usize,
    state: Mutex<ImportState<T, W>>,
}

impl<T, W> KVImport<T, W> {
    pub fn new(buffer: BufferId, request_pool_idx: usize, copy: bool) -> Self {
        Self {
            buffer,
            request_pool_idx,
            state: Mutex::new(ImportState {
                reads: Vec::new(),
                workspace: None,
                started: false,
                finished: !copy,
                drained: !copy,
                cancelled: false,
                released: false,
            }),
        }
    }

    pub fn cancelled(&self) -> bool {
        self.lock().cancelled
    }

    pub fn released(&self) -> bool {
        self.lock().released
    }

    pub fn start(&self) {
        self.lock().started = true;
    }

    /// A copy that could not drain its stream keeps all backing retained.
    pub fn finish(&self, drained: bool) {
        let mut state = self.lock();
        state.finished = true;
        state.drained = drained;
    }

    /// Cancelling a queued host task never enters its action. Its destination
    /// has no device access to drain, but still needs normal import retirement.
    pub fn task_done(&self) {
        let mut state = self.lock();
        if !state.started {
            state.finished = true;
            state.drained = true;
        }
    }

    pub fn workspace(&self) -> Option<Arc<W>> {
        self.lock().workspace.clone()
    }

    /// Register every started read, including one racing cancellation. The
    /// caller cancels it after unlocking when this returns true.
    pub fn retain(&self, read: Arc<T>) -> bool {
        let mut state = self.lock();
        state.reads.push(read);
        state.cancelled
    }

    pub fn visit<R>(&self, visit: impl FnOnce(&[Arc<T>], Option<&Arc<W>>) -> R) -> R {
        let state = self.lock();
        visit(&state.reads, state.workspace.as_ref())
    }

    fn lock(&self) -> MutexGuard<'_, ImportState<T, W>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

struct PoolState<I, W> {
    imports: HashMap<BufferId, I>,
    available: VecDeque<Arc<W>>,
    closed: bool,
}

/// Share a bounded set of conversion workspaces among native host tasks.
/// I retains a public import owner and dereferences its stable native state.
pub struct KVImporter<I, W> {
    state: Mutex<PoolState<I, W>>,
    available: Condvar,
}

impl<I, W> KVImporter<I, W> {
    pub fn new(workspaces: Vec<Arc<W>>) -> Self {
        Self {
            state: Mutex::new(PoolState {
                imports: HashMap::new(),
                available: workspaces.into(),
                closed: false,
            }),
            available: Condvar::new(),
        }
    }

    pub fn require_retired(&self) -> Result<()> {
        if !self.lock().imports.is_empty() {
            return Err(Error::Resource("KV imports still own physical storage"));
        }
        Ok(())
    }

    pub fn visit<R>(
        &self,
        visit: impl FnOnce(&HashMap<BufferId, I>, &VecDeque<Arc<W>>) -> R,
    ) -> Option<R> {
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => return None,
        };
        Some(visit(&state.imports, &state.available))
    }

    fn lock(&self) -> MutexGuard<'_, PoolState<I, W>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<T, I, W> KVImporter<I, W>
where
    I: Deref<Target = KVImport<T, W>>,
{
    /// Reserve native KV intervals and register their owner in one critical
    /// section. The reservation callback performs no foreign or device work.
    pub fn reserve(
        &self,
        write: I,
        reserve_destination: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        let mut state = self.lock();
        if state.closed || state.imports.contains_key(&write.buffer) {
            return Err(Error::Invalid(
                "KV import destination is closed or already reserved".into(),
            ));
        }
        reserve_destination()?;
        state.imports.insert(write.buffer, write);
        Ok(())
    }

    pub fn owns(&self, write: &KVImport<T, W>) -> bool {
        self.lock()
            .imports
            .get(&write.buffer)
            .is_some_and(|active| std::ptr::eq(&**active, write))
    }

    /// The backend first observes successful copy-task completion. Physical
    /// transport accesses may outlive that result and continue to hold pages.
    pub fn adopt(&self, write: &KVImport<T, W>) -> Result<()> {
        let state = self.lock();
        let owned = state
            .imports
            .get(&write.buffer)
            .is_some_and(|active| std::ptr::eq(&**active, write));
        let mut write = write.lock();
        if !owned || write.cancelled {
            return Err(Error::Invalid(
                "KV import destination is no longer active".into(),
            ));
        }
        write.released = true;
        Ok(())
    }

    /// Wait only on an import lane, never on the request execution thread.
    /// Cancellation and shutdown wake blocked workspace borrowers.
    pub fn acquire(&self, write: &KVImport<T, W>) -> Result<Arc<W>> {
        let mut state = self.lock();
        loop {
            if state.closed || write.cancelled() {
                return Err(Error::Resource("KV import was cancelled"));
            }
            if let Some(workspace) = state.available.pop_front() {
                write.lock().workspace = Some(Arc::clone(&workspace));
                return Ok(workspace);
            }
            state = self
                .available
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    pub fn require_active(&self, write: &KVImport<T, W>) -> Result<()> {
        if self.lock().closed || write.cancelled() {
            return Err(Error::Resource("KV import was cancelled"));
        }
        Ok(())
    }

    pub fn abandon(&self, write: &KVImport<T, W>) -> Vec<Arc<T>> {
        let state = self.lock();
        let reads = if state
            .imports
            .get(&write.buffer)
            .is_some_and(|active| std::ptr::eq(&**active, write))
        {
            cancel(write)
        } else {
            Vec::new()
        };
        drop(state);
        self.available.notify_all();
        reads
    }

    pub fn release(&self, buffers: &HashSet<BufferId>) -> Vec<Arc<T>> {
        self.cancel_matching(|buffer| buffers.contains(&buffer), false)
    }

    pub fn cancel_requests(
        &self,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Vec<Arc<T>> {
        self.cancel_matching(
            |buffer| requests.contains(&buffer.owner) && !retained.contains(&buffer),
            false,
        )
    }

    pub fn stop(&self) -> Vec<Arc<T>> {
        self.cancel_matching(|_| true, true)
    }

    fn cancel_matching(&self, selected: impl Fn(BufferId) -> bool, close: bool) -> Vec<Arc<T>> {
        let mut state = self.lock();
        state.closed |= close;
        let reads = state
            .imports
            .iter()
            .filter(|(buffer, _)| selected(**buffer))
            .flat_map(|(_, write)| cancel(write))
            .collect();
        drop(state);
        self.available.notify_all();
        reads
    }

    /// Return retired owners for completion notification after unlocking.
    /// Failed or cancelled physical completions continue to retain their reads.
    pub fn reap<E, C>(&self) -> Vec<I>
    where
        T: Deref<Target = Completion<E, C>>,
    {
        let mut retired_reads = Vec::new();
        let mut retired = Vec::new();
        {
            let mut state = self.lock();
            let mut buffers = Vec::new();
            let mut workspaces = Vec::new();
            for (&buffer, write) in &state.imports {
                let mut write = write.lock();
                let mut index = 0;
                while index < write.reads.len() {
                    if write.reads[index].succeeded() {
                        retired_reads.push(write.reads.swap_remove(index));
                    } else {
                        index += 1;
                    }
                }
                if !write.finished || !write.drained || !write.reads.is_empty() {
                    continue;
                }
                if let Some(workspace) = write.workspace.take() {
                    workspaces.push(workspace);
                }
                if write.released {
                    buffers.push(buffer);
                }
            }
            state.available.extend(workspaces);
            for buffer in buffers {
                if let Some(write) = state.imports.remove(&buffer) {
                    retired.push(write);
                }
            }
        }
        self.available.notify_all();
        drop(retired_reads);
        retired
    }
}

fn cancel<T, W>(write: &KVImport<T, W>) -> Vec<Arc<T>> {
    let mut state = write.lock();
    state.cancelled = true;
    state.released = true;
    state.reads.clone()
}
