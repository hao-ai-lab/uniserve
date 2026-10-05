//! Bounded host output storage and per-batch readback lifetimes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use uniserve_core::TokenLogprob;

use crate::cuda::{DeviceGuard, Event};
use crate::{Error, EventPool, Result};

mod logprobs;

pub use logprobs::LogprobLayout;

/// The numerical backend supplies the host allocation. Only storage is reused;
/// each batch receives a distinct OutputBuffer and completion signal.
pub struct OutputStorage<A> {
    pub value: A,
    pub words: usize,
    pub pinned: bool,
}

struct DeviceEvents {
    device: i32,
    start: Option<Arc<Event>>,
    copy: Option<Arc<Event>>,
    end: Option<Arc<Event>>,
}

/// A batch's readback ranges, device fences and row/CPU-reader lifetime.
/// Mutations are serialized by its owner; no numerical backend is required to
/// query CUDA completion or decide whether the allocation can be recycled.
pub struct OutputBuffer<A> {
    storage: Option<OutputStorage<A>>,
    rows: Vec<bool>,
    remaining: usize,
    tokens: usize,
    readback: Option<Vec<i64>>,
    logprob_layouts: HashMap<(usize, usize), LogprobLayout>,
    logprobs: HashMap<(usize, usize), HashMap<usize, Vec<TokenLogprob>>>,
    bytes: usize,
    devices: Vec<DeviceEvents>,
    sealed: bool,
    copies_finished: bool,
    abandoned: bool,
    readers: usize,
    timing_events: bool,
    reserved: Instant,
    started: Option<Instant>,
    copying: Option<Instant>,
    sealed_at: Option<Instant>,
    ready_at: Option<Instant>,
    timing: Option<[u64; 4]>,
}

impl<A> OutputBuffer<A> {
    pub fn new(storage: OutputStorage<A>, rows: usize, devices: &[i32], timing: bool) -> Self {
        Self {
            storage: Some(storage),
            rows: vec![false; rows],
            remaining: rows,
            tokens: 0,
            readback: None,
            logprob_layouts: HashMap::new(),
            logprobs: HashMap::new(),
            bytes: 0,
            devices: devices
                .iter()
                .map(|&device| DeviceEvents {
                    device,
                    start: None,
                    copy: None,
                    end: None,
                })
                .collect(),
            sealed: false,
            copies_finished: false,
            abandoned: false,
            readers: 0,
            timing_events: timing,
            reserved: Instant::now(),
            started: None,
            copying: None,
            sealed_at: None,
            ready_at: None,
            timing: None,
        }
    }

    pub fn sealed(&self) -> bool {
        self.sealed
    }

    pub fn storage(&self) -> Result<&OutputStorage<A>> {
        self.storage
            .as_ref()
            .ok_or_else(|| Error::Invariant("output storage has retired".into()))
    }

    /// Reserve opposing word and byte ranges before submitting a copy. A failed
    /// copy abandons its batch, so its range must remain occupied until retirement.
    pub fn reserve_tokens(&mut self, count: usize) -> Result<usize> {
        self.require_open()?;
        let available = (self.storage()?.words * 8 - self.bytes) / 8;
        let end = self
            .tokens
            .checked_add(count)
            .filter(|&end| end <= available)
            .ok_or(Error::Resource(
                "completion capture exceeds pinned output capacity",
            ))?;
        let offset = self.tokens;
        self.tokens = end;
        Ok(offset)
    }

    pub fn reserve_bytes(&mut self, count: usize) -> Result<usize> {
        self.require_open()?;
        let capacity = self.storage()?.words * 8;
        self.bytes = self
            .bytes
            .checked_add(count)
            .filter(|&end| count > 0 && end <= capacity - self.tokens * 8)
            .ok_or(Error::Resource(
                "byte capture exceeds pinned output capacity",
            ))?;
        Ok(capacity - self.bytes)
    }

    pub fn token_range(&self, offset: usize, count: usize) -> Result<()> {
        if !offset
            .checked_add(count)
            .is_some_and(|end| end <= self.tokens)
        {
            return Err(Error::Invariant(
                "completion capture range is outside its registered token extent".into(),
            ));
        }
        Ok(())
    }

    /// Snapshot completed words once, before the pinned allocation can be
    /// recycled. The backend reads exactly `count` initialized int64 words;
    /// byte captures retain their separate CPU-reader lifetime.
    pub fn readback(&mut self, read: impl FnOnce(&A, usize) -> Vec<i64>) -> Result<()> {
        if self.readback.is_some() {
            return Ok(());
        }
        if self.ready_at()?.is_none() {
            return Err(Error::Invariant(
                "completion storage was observed before its copy event was ready".into(),
            ));
        }

        self.readback = Some(read(&self.storage()?.value, self.tokens));
        Ok(())
    }

    pub fn read_tokens(&self, offset: usize, count: usize) -> Result<&[i64]> {
        self.token_range(offset, count)?;
        self.readback
            .as_ref()
            .and_then(|words| words.get(offset..offset + count))
            .ok_or(Error::State("completion words have not been read back"))
    }

    pub fn register_logprobs(
        &mut self,
        offset: usize,
        count: usize,
        layout: LogprobLayout,
    ) -> Result<()> {
        self.require_open()?;
        self.token_range(offset, count)?;
        layout.validate(count)?;
        self.logprob_layouts.insert((offset, count), layout);
        Ok(())
    }

    /// Upper bound before token deduplication, used for output-byte admission.
    pub fn logprob_entries(&self, span: (usize, usize, usize)) -> Result<usize> {
        let (offset, count, row) = span;
        self.logprob_layouts
            .get(&(offset, count))
            .ok_or(Error::State("logprob capture has no row layout"))?
            .entries(row)
    }

    /// IPC encodes a 4-byte row length and 12 bytes per token score. Admission
    /// uses the count before deduplication, so selected/top/requested overlap
    /// can only reduce the eventual payload.
    pub fn logprob_bytes(&self, spans: &[(usize, usize, usize)]) -> Result<usize> {
        spans.iter().try_fold(0, |bytes, &span| {
            Ok(bytes + 4 + 12 * self.logprob_entries(span)?)
        })
    }

    /// Selected token first, then top and explicitly requested tokens. The
    /// packed column is decoded once for all of its consuming request rows.
    pub fn logprob_values(&mut self, span: (usize, usize, usize)) -> Result<&[TokenLogprob]> {
        let (offset, count, row) = span;
        let key = (offset, count);
        if !self.logprobs.contains_key(&key) {
            let layout = self
                .logprob_layouts
                .get(&key)
                .ok_or(Error::State("logprob capture has no row layout"))?;
            let values = layout.decode(self.read_tokens(offset, count)?)?;
            self.logprobs.insert(key, values);
        }
        self.logprobs[&key]
            .get(&row)
            .map(Vec::as_slice)
            .ok_or(Error::State("logprob capture has no requested row"))
    }

    pub fn register_device(&self, device: Option<i32>) -> Result<()> {
        self.require_open()?;
        if device.is_some_and(|device| !self.devices.iter().any(|events| events.device == device)) {
            return Err(Error::Invariant(
                "output capture uses an undeclared CUDA device".into(),
            ));
        }
        Ok(())
    }

    pub fn begin_device<O>(
        &mut self,
        device: Option<i32>,
        stream: usize,
        pool: &mut EventPool<O>,
    ) -> Result<()> {
        self.register_device(device)?;
        self.started.get_or_insert_with(Instant::now);

        if let Some(device) = device.filter(|_| self.timing_events)
            && let Some(events) = self
                .devices
                .iter_mut()
                .find(|events| events.device == device)
            && events.start.is_none()
        {
            events.start = Some(record(pool, device, stream, true)?);
        }
        Ok(())
    }

    pub fn begin_copy<O>(
        &mut self,
        device: Option<i32>,
        stream: usize,
        pool: &mut EventPool<O>,
    ) -> Result<()> {
        self.register_device(device)?;
        self.copying.get_or_insert_with(Instant::now);

        if let Some(device) = device.filter(|_| self.timing_events) {
            self.begin_device(Some(device), stream, pool)?;
            if let Some(events) = self
                .devices
                .iter_mut()
                .find(|events| events.device == device)
                && events.copy.is_none()
            {
                events.copy = Some(record(pool, device, stream, true)?);
            }
        }
        Ok(())
    }

    /// Record all producer fences once. The caller transfers their pool
    /// references to deferred release and calls copies_finished after it drains.
    pub fn seal<O>(
        &mut self,
        streams: &[(i32, usize)],
        pool: &mut EventPool<O>,
    ) -> Result<Option<Vec<Arc<Event>>>> {
        if self.sealed {
            return Ok(None);
        }

        for &(device, stream) in streams {
            if self.timing_events {
                self.begin_copy(Some(device), stream, pool)?;
            }
            let events = self
                .devices
                .iter_mut()
                .find(|events| events.device == device)
                .ok_or_else(|| {
                    Error::Invariant("output fence uses an undeclared CUDA device".into())
                })?;
            if events.end.is_none() {
                events.end = Some(record(pool, device, stream, self.timing_events)?);
            }
        }

        self.sealed = true;
        self.sealed_at = Some(Instant::now());
        Ok(Some(self.events()))
    }

    pub fn events(&self) -> Vec<Arc<Event>> {
        self.devices
            .iter()
            .flat_map(|events| [&events.start, &events.copy, &events.end])
            .filter_map(|event| event.clone())
            .collect()
    }

    pub fn fences(&self) -> impl Iterator<Item = &Arc<Event>> {
        self.devices.iter().filter_map(|events| events.end.as_ref())
    }

    /// First observed completion time, or None while device writes are pending.
    pub fn ready_at(&mut self) -> Result<Option<Instant>> {
        if !self.sealed {
            return Ok(None);
        }
        if self.ready_at.is_some() {
            return Ok(self.ready_at);
        }

        for event in self.fences() {
            if !event.ready().map_err(Error::Cuda)? {
                return Ok(None);
            }
        }

        self.ready_at = Some(Instant::now());
        Ok(self.ready_at)
    }

    pub fn copies_finished(&mut self) {
        self.copies_finished = true;
        self.ready_at.get_or_insert_with(Instant::now);
    }

    pub fn observe(&mut self, row: usize) -> Result<[u64; 4]> {
        if row >= self.rows.len() {
            return Err(Error::Invariant(
                "completion row is outside its output buffer".into(),
            ));
        }
        let Some(ready_at) = self.ready_at()? else {
            return Err(Error::Invariant(
                "completion row was observed before query-ready".into(),
            ));
        };
        let timing = match self.timing {
            Some(timing) => timing,
            None => self.measure(ready_at)?,
        };
        self.timing = Some(timing);
        self.discard(row);
        Ok(timing)
    }

    pub fn timing(&self) -> Result<[u64; 4]> {
        self.timing
            .ok_or_else(|| Error::Invariant("completion timing was read before observation".into()))
    }

    pub fn discard(&mut self, row: usize) {
        if let Some(observed) = self.rows.get_mut(row)
            && !std::mem::replace(observed, true)
        {
            self.remaining -= 1;
        }
    }

    pub fn abandon(&mut self) {
        self.abandoned = true;
    }

    pub fn retain_reader(&mut self) -> Result<()> {
        self.storage()?;
        self.readers += 1;
        Ok(())
    }

    pub fn release_reader(&mut self) -> Result<()> {
        self.readers = self
            .readers
            .checked_sub(1)
            .ok_or_else(|| Error::Invariant("output CPU reader count underflow".into()))?;
        Ok(())
    }

    fn retired(&self) -> bool {
        self.copies_finished && (self.remaining == 0 || self.abandoned) && self.readers == 0
    }

    fn require_open(&self) -> Result<()> {
        if self.sealed {
            return Err(Error::Invariant("output capture is already sealed".into()));
        }
        Ok(())
    }

    fn measure(&self, ready_at: Instant) -> Result<[u64; 4]> {
        let queued = self.started.map_or(0, |start| micros(start, self.reserved));
        let mut device = 0;
        let mut copy = 0;
        if self.timing_events {
            for events in &self.devices {
                if let (Some(start), Some(producer), Some(end)) =
                    (&events.start, &events.copy, &events.end)
                {
                    // CUDA reports milliseconds; output statistics use microseconds.
                    device = device.max(
                        (start.elapsed_time(producer).map_err(Error::Cuda)? as f64 * 1000.0)
                            .round_ties_even() as u64,
                    );
                    copy = copy.max(
                        (producer.elapsed_time(end).map_err(Error::Cuda)? as f64 * 1000.0)
                            .round_ties_even() as u64,
                    );
                }
            }
        }
        if self.devices.is_empty()
            && let (Some(start), Some(sealed)) = (self.started, self.sealed_at)
        {
            let copying = self.copying.unwrap_or(sealed);
            device = micros(copying, start);
            copy = micros(sealed, copying);
        }
        Ok([queued, device, copy, micros(Instant::now(), ready_at)])
    }
}

fn micros(end: Instant, start: Instant) -> u64 {
    end.saturating_duration_since(start).as_micros() as u64
}

fn record<O>(
    pool: &mut EventPool<O>,
    device: i32,
    stream: usize,
    timing: bool,
) -> Result<Arc<Event>> {
    // Default-stream handles do not identify a device. Select its context
    // before the driver resolves that stream and records the fence.
    let _device = DeviceGuard::new(device).map_err(Error::Cuda)?;
    let event = pool.acquire(device, timing, false)?;
    pool.retain(&event, device, 1)?;
    pool.record(&event, device, stream)?;
    Ok(event)
}

type Lease<A> = Arc<Mutex<OutputBuffer<A>>>;

/// Owners retain language-side views until their lease retires. Reaping returns
/// those owners for destruction outside the pool lock; it recycles only storage.
pub struct OutputPool<A, O> {
    capacity: usize,
    max_words: usize,
    active: Vec<(Lease<A>, O)>,
    available: Vec<OutputStorage<A>>,
    closed: bool,
}

impl<A, O> OutputPool<A, O> {
    pub fn new(capacity: usize, max_words: usize) -> Self {
        Self {
            capacity,
            max_words,
            active: Vec::new(),
            available: Vec::new(),
            closed: false,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn max_words(&self) -> usize {
        self.max_words
    }

    pub fn reap(&mut self) -> Vec<O> {
        let mut retired = Vec::new();
        let mut index = 0;
        while index < self.active.len() {
            let mut buffer = self.active[index]
                .0
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if !buffer.retired() {
                index += 1;
                continue;
            }
            if let Some(storage) = buffer.storage.take() {
                self.available.push(storage);
            }
            // Released event objects can now be reused even if a result retains
            // its consumed lease. Timing and decoded values remain per lease.
            buffer.devices.clear();
            drop(buffer);
            retired.push(self.active.swap_remove(index).1);
        }
        retired
    }

    /// The caller serializes taking storage, numerical allocation and insertion.
    pub fn take(&mut self, rows: usize, words: usize) -> Result<Option<OutputStorage<A>>> {
        if self.closed {
            return Err(Error::Resource("output pool is closed"));
        }
        if words > self.max_words {
            return Err(Error::Resource(
                "lane output exceeds its startup storage bound",
            ));
        }
        if rows == 0 || words < rows {
            return Err(Error::Resource(
                "completion row count exceeds pinned output capacity",
            ));
        }
        if self.active.len() >= self.capacity {
            return Err(Error::Resource("all lane output leases are active"));
        }
        Ok(self.available.pop())
    }

    pub fn insert(&mut self, buffer: Lease<A>, owner: O) {
        self.active.push((buffer, owner));
    }

    pub fn owners(&self) -> impl Iterator<Item = &O> {
        self.active.iter().map(|(_, owner)| owner)
    }

    pub fn storages(&self) -> impl Iterator<Item = &OutputStorage<A>> {
        self.available.iter()
    }

    pub fn begin_close(&mut self) {
        self.closed = true;
    }

    /// Release owners only after shutdown has drained their device writes.
    /// A failed drain leaves them retained for the caller's error handling.
    pub fn finish_close(&mut self) -> (Vec<O>, Vec<OutputStorage<A>>) {
        (
            self.active.drain(..).map(|(_, owner)| owner).collect(),
            std::mem::take(&mut self.available),
        )
    }
}
