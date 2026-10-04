//! Bounded host execution shared by media encoding and cache transfer.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Receiver, Sender, unbounded};

use crate::{Completion, Error, Outcome, Result};

/// Numerical work and language-runtime observers supplied to a host task.
/// The lane owns admission, scheduling, results, and input retirement.
pub trait HostAction: Send + Sync + Sized + 'static {
    type Output: Send + Sync;
    type Error: Send + Sync;
    type Callback: Send;
    type Wake: Send + Sync;

    fn ready(&self) -> std::result::Result<bool, Self::Error>;
    fn run(&self) -> std::result::Result<Self::Output, Self::Error>;
    fn release(&self) -> std::result::Result<(), Self::Error>;
    fn input_outcome(&self) -> Option<Outcome<Self::Error>>;

    /// Arrange one call to `retire_input` when the producer completes.
    fn defer_release(task: Arc<HostTask<Self>>) -> std::result::Result<(), Self::Error>;
    fn notify(callbacks: Vec<Self::Callback>);
    fn wake(wake: &Self::Wake);
    fn report(error: Self::Error);
    fn error(error: Error) -> Self::Error;
    fn note_cleanup(error: &mut Self::Error, cleanup: Self::Error);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Reserved,
    Queued,
    Running,
    Finished,
}

struct TaskState<A> {
    phase: Phase,
    action: Option<Arc<A>>,
}

/// One admitted action and its result. Queued cancellation resolves the result
/// immediately, but keeps admission capacity until a worker dequeues the task.
pub struct HostTask<A: HostAction> {
    lane: Weak<Mutex<LaneState<A>>>,
    state: Mutex<TaskState<A>>,
    pub completion: Completion<A::Error, A::Callback, Arc<A::Output>>,
}

impl<A: HostAction> HostTask<A> {
    fn state(&self) -> MutexGuard<'_, TaskState<A>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn action(&self) -> Option<Arc<A>> {
        self.state().action.clone()
    }

    pub fn configure(&self, action: A) -> Result<()> {
        let lane = self
            .lane
            .upgrade()
            .ok_or(Error::State("host task is no longer admitted"))?;
        let lane = lane.lock().unwrap_or_else(PoisonError::into_inner);
        let mut state = self.state();

        if lane.closed || state.phase != Phase::Reserved {
            return Err(Error::State("host task is no longer admitted"));
        }
        if state.action.is_some() {
            return Err(Error::State("host task was configured more than once"));
        }

        state.action = Some(Arc::new(action));
        Ok(())
    }

    pub fn submit(self: &Arc<Self>) -> Result<()> {
        let lane = self
            .lane
            .upgrade()
            .ok_or(Error::State("host task is no longer admitted"))?;
        let mut lane = lane.lock().unwrap_or_else(PoisonError::into_inner);
        let mut state = self.state();

        if lane.closed || state.phase == Phase::Finished {
            return Err(Error::State("host task is no longer admitted"));
        }
        if state.phase != Phase::Reserved {
            return Err(Error::State("host task was submitted more than once"));
        }
        if state.action.is_none() {
            return Err(Error::State("host task has no action"));
        }

        let worker = lane
            .workers
            .iter_mut()
            .min_by_key(|worker| worker.queued)
            .ok_or(Error::State("host lane has no workers"))?;

        // Sending under the admission lock orders accepted work before close's
        // stop messages. The channel is unbounded; admission supplies its bound.
        worker
            .queue
            .send(Some(Arc::clone(self)))
            .map_err(|_| Error::State("host worker has stopped"))?;
        worker.queued += 1;
        state.phase = Phase::Queued;

        Ok(())
    }

    pub fn submit_if_ready(self: &Arc<Self>) -> std::result::Result<(), A::Error> {
        let action = {
            let state = self.state();
            if state.phase != Phase::Reserved {
                return Ok(());
            }
            state.action.clone()
        }
        .ok_or_else(|| A::error(Error::State("host task has no action")))?;

        if action.ready()? && !self.completion.done() {
            self.submit().map_err(A::error)?;
        }
        Ok(())
    }

    /// Abandon withdraws only unsubmitted work; cancel also withdraws queued
    /// work. Running actions keep their capacity and input leases to completion.
    pub fn cancel(self: &Arc<Self>, abandon: bool) -> std::result::Result<bool, A::Error> {
        let lane = self.lane.upgrade();
        let mut lane = lane
            .as_ref()
            .map(|lane| lane.lock().unwrap_or_else(PoisonError::into_inner));
        let mut state = self.state();

        if matches!(state.phase, Phase::Running | Phase::Finished)
            || (abandon && state.phase == Phase::Queued)
        {
            return Ok(false);
        }

        let unused = state.phase == Phase::Reserved;
        if unused {
            state.phase = Phase::Finished;
            if let Some(lane) = lane.as_mut() {
                lane.tasks.remove(&(Arc::as_ptr(self) as usize));
            }
        }
        let (cancelled, callbacks) = self.completion.cancel();
        drop(state);
        drop(lane);

        let released = if unused { self.release_input() } else { Ok(()) };
        A::notify(callbacks);
        released.map(|()| cancelled)
    }

    /// Release only after the physical producer has stopped. Failed completion
    /// retains its input for process teardown, independently of task results.
    #[expect(
        clippy::mem_forget,
        reason = "unknown device completion cannot release a borrowed input"
    )]
    pub fn retire_input(&self) -> std::result::Result<bool, A::Error> {
        let Some(action) = self.action() else {
            return Ok(true);
        };
        let Some(outcome) = action.input_outcome() else {
            return Ok(false);
        };

        let action = self.state().action.take();
        let Some(action) = action else {
            return Ok(true);
        };

        match outcome {
            Outcome::Success(()) => action.release()?,
            Outcome::Failed(_) | Outcome::Cancelled => std::mem::forget(action),
        }

        Ok(true)
    }

    fn release_input(self: &Arc<Self>) -> std::result::Result<(), A::Error> {
        if !self.retire_input()? {
            A::defer_release(Arc::clone(self))?;
        }
        Ok(())
    }

    #[expect(
        clippy::expect_used,
        reason = "only configured queued tasks reach a worker; running tasks reject cancellation"
    )]
    fn run(self: &Arc<Self>, aborted: bool) {
        let (action, cancelled, callbacks) = {
            let mut state = self.state();
            let callbacks = if aborted {
                self.completion.cancel().1
            } else {
                Vec::new()
            };
            let cancelled = self.completion.done();
            state.phase = if cancelled {
                Phase::Finished
            } else {
                Phase::Running
            };
            (
                state.action.clone().expect("configured host task"),
                cancelled,
                callbacks,
            )
        };
        A::notify(callbacks);

        if cancelled {
            if let Err(error) = self.release_input() {
                A::report(error);
            }
            self.remove();
            return;
        }

        let mut result = action.run();
        if let Err(cleanup) = self.release_input() {
            match &mut result {
                Ok(_) => result = Err(cleanup),
                Err(error) => A::note_cleanup(error, cleanup),
            }
        }
        self.state().phase = Phase::Finished;

        // A result observer can immediately reserve the returned capacity.
        let wake = self.remove();
        let callbacks = self
            .completion
            .complete(result.map(Arc::new))
            .expect("unfinished running task");
        A::notify(callbacks);
        if let Some(wake) = wake {
            A::wake(&wake);
        }
    }

    fn remove(self: &Arc<Self>) -> Option<Arc<A::Wake>> {
        let lane = self.lane.upgrade()?;
        let mut lane = lane.lock().unwrap_or_else(PoisonError::into_inner);
        lane.tasks.remove(&(Arc::as_ptr(self) as usize));
        lane.wake.clone()
    }

    /// Visit an unshared action for language-runtime garbage collection.
    /// A worker borrowing the action is an external owner of its references.
    pub fn visit_action<R>(&self, visitor: impl FnOnce(&A) -> R) -> Option<R> {
        let state = self.state();
        state
            .action
            .as_ref()
            .filter(|action| Arc::strong_count(action) == 1)
            .map(|action| visitor(action))
    }

    /// Drop references from an unreachable, uniquely owned language wrapper.
    pub fn clear(&mut self) {
        self.state = Mutex::new(TaskState {
            phase: Phase::Finished,
            action: None,
        });
        self.completion = Completion::default();
    }
}

struct Worker<A: HostAction> {
    queue: Sender<Option<Arc<HostTask<A>>>>,
    queued: usize,
}

struct LaneState<A: HostAction> {
    capacity: usize,
    closed: bool,
    aborted: bool,
    tasks: HashMap<usize, Arc<HostTask<A>>>,
    workers: Vec<Worker<A>>,
    wake: Option<Arc<A::Wake>>,
}

/// A bounded lane with native worker threads. Abort stops admission without
/// joining tasks that may be waiting on a failed device or peer.
pub struct HostLane<A: HostAction> {
    state: Arc<Mutex<LaneState<A>>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl<A: HostAction> HostLane<A> {
    pub fn new(capacity: usize, workers: usize) -> Result<Self> {
        if capacity == 0 || workers == 0 || workers > capacity {
            return Err(Error::Invalid(
                "host workers must lie within positive lane capacity".into(),
            ));
        }
        let mut queues = Vec::new();
        let mut receivers = Vec::new();
        for _ in 0..workers {
            let (queue, receiver) = unbounded();
            queues.push(Worker { queue, queued: 0 });
            receivers.push(receiver);
        }
        let state = Arc::new(Mutex::new(LaneState {
            capacity,
            closed: false,
            aborted: false,
            tasks: HashMap::new(),
            workers: queues,
            wake: None,
        }));
        let mut threads = Vec::new();
        for (index, receiver) in receivers.into_iter().enumerate() {
            let shared = Arc::clone(&state);
            match thread::Builder::new()
                .name(format!("worker-host-lane-{index}"))
                .spawn(move || run_worker(shared, index, receiver))
            {
                Ok(thread) => threads.push(thread),
                Err(error) => {
                    for worker in &state.lock().unwrap_or_else(PoisonError::into_inner).workers {
                        let _ = worker.queue.send(None);
                    }
                    for thread in threads {
                        let _ = thread.join();
                    }
                    return Err(Error::Invariant(format!(
                        "cannot start host worker: {error}"
                    )));
                }
            }
        }
        Ok(Self {
            state,
            threads: Mutex::new(threads),
        })
    }

    fn state(&self) -> MutexGuard<'_, LaneState<A>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn reserved(&self) -> usize {
        self.state().tasks.len()
    }

    pub fn set_wake(&self, wake: Option<A::Wake>) {
        let previous = std::mem::replace(&mut self.state().wake, wake.map(Arc::new));
        drop(previous);
    }

    pub fn reserve(&self) -> Result<Arc<HostTask<A>>> {
        let mut state = self.state();
        if state.closed {
            return Err(Error::Resource("worker host lane is closed"));
        }
        if state.tasks.len() >= state.capacity {
            return Err(Error::Resource("worker host lane capacity is exhausted"));
        }
        let task = Arc::new(HostTask {
            lane: Arc::downgrade(&self.state),
            state: Mutex::new(TaskState {
                phase: Phase::Reserved,
                action: None,
            }),
            completion: Completion::default(),
        });
        state
            .tasks
            .insert(Arc::as_ptr(&task) as usize, Arc::clone(&task));
        Ok(task)
    }

    pub fn abort(&self) {
        let wake = {
            let mut state = self.state();
            state.closed = true;
            state.aborted = true;
            for worker in &state.workers {
                let _ = worker.queue.send(None);
            }
            state.wake.take()
        };
        drop(wake);
    }

    pub fn close(&self) -> Vec<A::Error> {
        let unused: Vec<_> = {
            let mut state = self.state();
            state.closed = true;
            state
                .tasks
                .values()
                .filter(|task| task.state().phase == Phase::Reserved)
                .cloned()
                .collect()
        };
        let mut errors = Vec::new();
        for task in unused {
            if let Err(error) = task.cancel(true) {
                errors.push(error);
            }
        }
        {
            let state = self.state();
            for worker in &state.workers {
                let _ = worker.queue.send(None);
            }
        }
        let threads =
            std::mem::take(&mut *self.threads.lock().unwrap_or_else(PoisonError::into_inner));
        for thread in threads {
            if thread.thread().id() == thread::current().id() {
                errors.push(A::error(Error::State("host worker cannot join itself")));
                continue;
            }
            if thread.join().is_err() {
                errors.push(A::error(Error::State("host worker panicked")));
            }
        }
        self.set_wake(None);
        errors
    }
}

impl<A: HostAction> Drop for HostLane<A> {
    fn drop(&mut self) {
        // A destructor must not wait on a failed peer or require the GIL.
        self.abort();
    }
}

fn run_worker<A: HostAction>(
    state: Arc<Mutex<LaneState<A>>>,
    index: usize,
    receiver: Receiver<Option<Arc<HostTask<A>>>>,
) {
    while let Ok(Some(task)) = receiver.recv() {
        let aborted = state.lock().unwrap_or_else(PoisonError::into_inner).aborted;
        task.run(aborted);
        state.lock().unwrap_or_else(PoisonError::into_inner).workers[index].queued -= 1;
    }
}
