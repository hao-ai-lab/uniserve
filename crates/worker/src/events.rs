//! Shared CUDA event leases and deferred storage release.

use std::collections::HashMap;
use std::sync::Arc;

use crate::cuda::{Event, Stream};
use crate::{Error, Result};

type PoolKey = (i32, bool, bool);

#[derive(PartialEq)]
enum Status {
    Open,
    Closing,
    Closed,
}

struct ActiveEvent {
    event: Arc<Event>,
    references: usize,
    stream: Option<usize>,
    recorded: bool,
}

struct DeferredRelease<O> {
    events: Vec<Arc<Event>>,
    owner: O,
}

/// Owns pooled events until their references and device accesses retire.
///
/// The owner serializes mutations. Completed deferred owners are returned to
/// it, so callbacks and destruction can run after releasing the ownership lock.
pub struct EventPool<O> {
    available: HashMap<PoolKey, Vec<Arc<Event>>>,
    active: HashMap<usize, ActiveEvent>,
    deferred: Vec<DeferredRelease<O>>,
    wake_streams: HashMap<(i32, usize), Arc<Stream>>,
    status: Status,
}

impl<O> Default for EventPool<O> {
    fn default() -> Self {
        Self {
            available: HashMap::new(),
            active: HashMap::new(),
            deferred: Vec::new(),
            wake_streams: HashMap::new(),
            status: Status::Open,
        }
    }
}

impl<O> EventPool<O> {
    /// Acquires an unrecorded lease with no retained references. Storage owners
    /// retain their references before transferring them to release or deferral.
    pub fn acquire(&mut self, device: i32, timing: bool, interprocess: bool) -> Result<Arc<Event>> {
        self.require_open()?;

        let available = self
            .available
            .entry((device, timing, interprocess))
            .or_default();
        // A caller may retain an event object after releasing its lease.
        // Reuse only when the pool is its last owner, preserving that observation.
        let reusable = available
            .iter()
            .position(|event| Arc::strong_count(event) == 1);
        let event = reusable
            .map(|index| available.swap_remove(index))
            .unwrap_or_else(|| Arc::new(Event::new(device, timing, interprocess)));
        self.active.insert(
            key(&event),
            ActiveEvent {
                event: Arc::clone(&event),
                references: 0,
                stream: None,
                recorded: false,
            },
        );
        Ok(event)
    }

    /// Binds the producer stream before a later record call submits the fence.
    pub fn declare_stream(
        &mut self,
        event: &Arc<Event>,
        device: i32,
        stream: usize,
    ) -> Result<usize> {
        self.require_open()?;
        let state = self.require(event, device)?;
        if state.stream.is_some_and(|previous| previous != stream) {
            return Err(invariant(
                "device event spans incompatible producer streams",
            ));
        }
        state.stream = Some(stream);
        Ok(stream)
    }

    /// Records this lease once, honoring its declared stream when present.
    pub fn record(&mut self, event: &Arc<Event>, device: i32, current: usize) -> Result<usize> {
        self.require_open()?;
        let state = self.require(event, device)?;
        if state.recorded {
            return Err(invariant("device event was recorded more than once"));
        }
        let stream = state.stream.unwrap_or(current);
        event.record(stream).map_err(Error::Cuda)?;
        state.stream = Some(stream);
        state.recorded = true;
        Ok(stream)
    }

    pub fn retain(&mut self, event: &Arc<Event>, device: i32, count: usize) -> Result<()> {
        self.require_open()?;
        if count == 0 {
            return Err(invariant("device event retain count must be positive"));
        }
        self.require(event, device)?.references += count;
        Ok(())
    }

    /// Returns references after their last device access. A rejected final
    /// release preserves its references so the caller can defer them instead.
    pub fn release(&mut self, event: &Arc<Event>, count: usize) -> Result<()> {
        // Shutdown already drained and released every lease. A completed
        // deferred owner's callback may still release another retained event.
        if self.status == Status::Closed {
            return Ok(());
        }
        let state = self.require(event, event.device())?;
        if count == 0 || state.references < count {
            return Err(invariant("device event reference accounting is invalid"));
        }
        if state.references == count && (!state.recorded || !event.ready().map_err(Error::Cuda)?) {
            return Err(invariant(
                "device event was released before it became query-ready",
            ));
        }

        self.release_ready(event, count);
        Ok(())
    }

    /// Transfers existing references and their storage owner to deferred release.
    /// An empty or already drained pool returns the owner without retaining it.
    pub fn defer_release(&mut self, events: Vec<Arc<Event>>, owner: O) -> Result<Option<O>> {
        if self.status == Status::Closed || events.is_empty() {
            return Ok(Some(owner));
        }
        self.require_references(&events)?;
        self.deferred.push(DeferredRelease { events, owner });
        Ok(None)
    }

    /// Returns retired owners and the first failure, retaining every failed
    /// dependency. Independent completed owners still retire after a failure.
    pub fn reap(&mut self) -> (Vec<O>, Result<()>) {
        let mut completed = Vec::new();
        let mut failure = Ok(());
        let mut index = 0;

        while index < self.deferred.len() {
            let deferred = &self.deferred[index];
            let ready = deferred.events.iter().try_fold(true, |ready, event| {
                event.ready().map(|done| ready && done).map_err(Error::Cuda)
            });
            let result = ready.and_then(|ready| {
                if ready {
                    self.require_references(&deferred.events)?;
                }
                Ok(ready)
            });

            match result {
                Ok(true) => {
                    let deferred = self.deferred.swap_remove(index);
                    for event in &deferred.events {
                        self.release_ready(event, 1);
                    }
                    completed.push(deferred.owner);
                }
                result => {
                    if let Err(error) = result
                        && failure.is_ok()
                    {
                        failure = Err(error);
                    }
                    index += 1;
                }
            }
        }
        (completed, failure)
    }

    /// Selects an independent notification stream for this producer and orders
    /// it after the event. The caller selects the device context for creation.
    pub fn wake_stream(&mut self, event: &Arc<Event>, device: i32) -> Result<Arc<Stream>> {
        self.require_open()?;
        let state = self.require(event, device)?;
        let producer = state.stream.filter(|_| state.recorded).ok_or_else(|| {
            invariant("completion notification requires a recorded producer event")
        })?;
        let stream = match self.wake_streams.get(&(device, producer)) {
            Some(stream) => Arc::clone(stream),
            None => {
                let stream = Arc::new(Stream::new().map_err(Error::Cuda)?);
                self.wake_streams
                    .insert((device, producer), Arc::clone(&stream));
                stream
            }
        };
        event.wait_on(stream.handle()).map_err(Error::Cuda)?;
        Ok(stream)
    }

    /// Stops new work and returns a device wait that runs outside the owner lock.
    /// Existing leases may still release or defer their retained storage.
    pub fn begin_close(&mut self) -> impl FnOnce() -> Result<()> + Send + 'static {
        if self.status == Status::Open {
            self.status = Status::Closing;
        }
        let streams: Vec<_> = self.wake_streams.values().cloned().collect();
        let events: Vec<_> = self
            .active
            .values()
            .filter(|state| state.recorded)
            .map(|state| Arc::clone(&state.event))
            .collect();

        move || {
            for stream in streams {
                stream.wait().map_err(Error::Cuda)?;
            }
            for event in events {
                event.wait().map_err(Error::Cuda)?;
            }
            Ok(())
        }
    }

    /// Releases the pool after the wait from begin_close succeeds.
    pub fn finish_close(&mut self) -> Vec<O> {
        self.status = Status::Closed;
        self.wake_streams.clear();
        self.active.clear();
        self.available.clear();
        self.deferred
            .drain(..)
            .map(|release| release.owner)
            .collect()
    }

    pub fn deferred_owners(&self) -> impl Iterator<Item = &O> {
        self.deferred.iter().map(|release| &release.owner)
    }

    fn require_open(&self) -> Result<()> {
        if self.status != Status::Open {
            return Err(invariant("device event pool is closed"));
        }
        Ok(())
    }

    fn require(&mut self, event: &Arc<Event>, device: i32) -> Result<&mut ActiveEvent> {
        if event.device() != device {
            return Err(invariant(
                "device event is not owned by its declared device",
            ));
        }
        self.active
            .get_mut(&key(event))
            .ok_or_else(|| invariant("device event is not owned by its declared device"))
    }

    fn require_references(&self, events: &[Arc<Event>]) -> Result<()> {
        let mut counts = HashMap::new();
        for event in events {
            *counts.entry(key(event)).or_insert(0_usize) += 1;
        }
        for (key, count) in counts {
            if self
                .active
                .get(&key)
                .is_none_or(|state| state.references < count)
            {
                return Err(invariant("deferred device event has invalid ownership"));
            }
        }
        Ok(())
    }

    fn release_ready(&mut self, event: &Arc<Event>, count: usize) {
        if let Some(state) = self.active.get_mut(&key(event)) {
            state.references -= count;
            if state.references == 0 {
                self.active.remove(&key(event));
                self.available
                    .entry((event.device(), event.timing(), event.interprocess()))
                    .or_default()
                    .push(Arc::clone(event));
            }
        }
    }
}

impl<O> Drop for EventPool<O> {
    #[allow(clippy::mem_forget)]
    fn drop(&mut self) {
        // A caller may drop an unclosed pool. Unknown physical completion must
        // retain deferred storage rather than return it to an allocator.
        if !self.deferred.is_empty() && self.begin_close()().is_err() {
            std::mem::forget(std::mem::take(&mut self.deferred));
        }
    }
}

fn key(event: &Arc<Event>) -> usize {
    Arc::as_ptr(event) as usize
}

fn invariant(message: &str) -> Error {
    Error::Invariant(message.into())
}
