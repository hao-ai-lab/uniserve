//! Cross-call tensor storage, read leases and physical retirement.
//!
//! The store serializes changes to its buffers. Numerical backends retain
//! buffer and import handles, and fixed arenas. Handles borrow native state
//! directly; their foreign owners never decide when storage can be reused.

use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use uniserve_core::CallId;
use uniserve_worker_ipc::{BufferId, RequestKey};

use crate::cuda::Event;
use crate::{BufferBinding, Completion, Error, Result};

pub type BufferKey = (RequestKey, CallId, u16);
pub type CallKey = (RequestKey, CallId);
pub type LaneKey = (String, usize, usize);
pub type RelayKey = (LaneKey, String, usize);
pub type ArenaKey = (String, String, usize);
type RelayCall = (String, usize, RequestKey, CallId);

pub fn buffer_key(id: BufferId) -> BufferKey {
    (id.owner, id.producer_call_id, id.output_index)
}

/// Production ends a numerical write; commit makes its value discoverable.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WriteState {
    Reserved,
    Deferred,
    Produced,
    Committed,
}

impl WriteState {
    pub fn produced(self) -> bool {
        matches!(self, Self::Produced | Self::Committed)
    }
}

pub enum Backing {
    Persistent(Arc<BufferBinding>),
    Relay(RelayKey),
}

/// A value's event and the stream on which its producer will record it.
#[derive(Clone)]
pub struct Fence {
    pub event: Arc<Event>,
    pub stream: usize,
}

/// One allocation-backed value. The owning store protects mutations, while
/// read handles retain the same buffer through logical release and reuse.
/// T and C retain transfers and exports and borrow their native completions.
pub struct Buffer<T, C> {
    pub id: BufferId,
    pub device: String,
    pub logical_shape: Vec<usize>,
    pub feature: bool,
    pub backing: Option<Backing>,
    pub state: WriteState,
    pub extent: usize,
    pub value_shape: Vec<usize>,
    pub producer: Option<Fence>,
    pub readers: usize,
    pub reader_events: Vec<Arc<Event>>,
    pub transfers: Vec<T>,
    pub exports: Vec<C>,
    pub released: bool,
}

impl<T, C> Buffer<T, C> {
    pub fn new(
        id: BufferId,
        device: String,
        logical_shape: Vec<usize>,
        feature: bool,
        backing: Backing,
    ) -> Self {
        Self {
            id,
            device,
            logical_shape,
            feature,
            backing: Some(backing),
            state: WriteState::Reserved,
            extent: 0,
            value_shape: Vec::new(),
            producer: None,
            readers: 0,
            reader_events: Vec::new(),
            transfers: Vec::new(),
            exports: Vec::new(),
            released: false,
        }
    }

    pub fn require_candidate(&self) -> Result<()> {
        if self.backing.is_none() || self.state == WriteState::Committed {
            return Err(Error::Invariant(
                "device-product candidate is not live".into(),
            ));
        }
        if self.state == WriteState::Reserved {
            return Err(Error::Invariant(
                "completion packing found an unpublished device product".into(),
            ));
        }
        Ok(())
    }

    pub fn defer(&mut self) -> Result<()> {
        if self.state.produced() {
            return Err(Error::Invariant(
                "a published write cannot be deferred".into(),
            ));
        }
        self.state = WriteState::Deferred;
        Ok(())
    }

    pub fn produced(&mut self, shape: Vec<usize>, extent: usize, fence: Option<Fence>) {
        self.value_shape = shape;
        self.extent = extent;
        self.producer = fence;
        self.state = WriteState::Produced;
    }

    pub fn retain_reader(&mut self, event: &Arc<Event>) -> bool {
        if self
            .reader_events
            .iter()
            .any(|current| Arc::ptr_eq(current, event))
        {
            return false;
        }
        self.reader_events.push(Arc::clone(event));
        true
    }

    pub fn events(&self) -> impl Iterator<Item = &Arc<Event>> {
        self.producer
            .iter()
            .map(|fence| &fence.event)
            .chain(&self.reader_events)
    }

    /// Cancellation is not physical completion. A failed transfer or export
    /// retains its backing until the worker's failure shutdown drains access.
    pub fn access_retired<TE, TC, CE, CC>(&self, queried: &mut HashMap<usize, bool>) -> Result<bool>
    where
        T: Deref<Target = Completion<TE, TC>>,
        C: Deref<Target = Completion<CE, CC>>,
    {
        if !self.released
            || self.readers != 0
            || self.transfers.iter().any(|transfer| !transfer.succeeded())
            || self.exports.iter().any(|export| !export.succeeded())
        {
            return Ok(false);
        }

        for event in self.events() {
            let ready = match queried.get(&(Arc::as_ptr(event) as usize)) {
                Some(&ready) => ready,
                None => {
                    let ready = event.ready().map_err(Error::Cuda)?;
                    queried.insert(Arc::as_ptr(event) as usize, ready);
                    ready
                }
            };
            if !ready {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn retain_export<E, F>(&mut self, retirement: C)
    where
        C: Deref<Target = Completion<E, F>>,
    {
        self.exports.retain(|export| !export.succeeded());
        self.exports.push(retirement);
    }

    pub fn retain_transfer(&mut self, ticket: T) -> Result<()> {
        if self.state.produced() || self.released {
            return Err(Error::Invariant(
                "transfer destination already has a producer".into(),
            ));
        }
        self.transfers.push(ticket);
        Ok(())
    }
}

/// Missing coverage materialized once and shared by its concurrent readers.
pub struct TensorImport<B, T> {
    pub buffer: B,
    pub tickets: Vec<T>,
    pub users: usize,
    pub committed: bool,
}

impl<B, T> TensorImport<B, T> {
    pub fn new(buffer: B, tickets: Vec<T>, committed: bool) -> Self {
        Self {
            buffer,
            tickets,
            users: 1,
            committed,
        }
    }

    pub fn share<C>(&mut self)
    where
        B: Deref<Target = Mutex<Buffer<T, C>>>,
    {
        lock(&self.buffer).readers += 1;
        self.users += 1;
    }
}

/// One consumer's access. The numerical caller keeps a stable tensor view
/// even when another import extends the buffer's resident coverage.
pub struct TensorRead<B, I> {
    pub buffer: B,
    pub consumer: Option<CallId>,
    pub imported: Option<I>,
    pub complete: bool,
}

impl<B, I> TensorRead<B, I> {
    pub fn complete<T, C>(&mut self) -> Option<I>
    where
        B: Deref<Target = Mutex<Buffer<T, C>>>,
        I: Deref<Target = Mutex<TensorImport<B, T>>>,
    {
        if self.complete {
            return None;
        }
        self.complete = true;
        lock(&self.buffer).readers -= 1;

        let imported = self.imported.take()?;
        let mut import = lock(&imported);
        import.users -= 1;
        let last = import.users == 0;
        if last && !lock(&self.buffer).state.produced() {
            lock(&self.buffer).released = true;
        }
        drop(import);
        last.then_some(imported)
    }
}

/// A call holds its relay lane until all scalar fields have retired.
struct Lane {
    call: CallKey,
    fields: HashSet<(String, usize)>,
}

/// Rank-local values and bounded scalar relay lanes. B and I retain the
/// numerical owners of native buffers and imports; A holds an arena or view.
pub struct TensorStore<B, I, A> {
    pub capacity: usize,
    pub byte_capacity: usize,
    pub request_capacity: usize,
    pub relay_depth: usize,
    pub buffers: HashMap<BufferKey, B>,
    pub imports: HashMap<BufferKey, I>,
    pub arenas: HashMap<ArenaKey, A>,
    pub views: HashMap<RelayKey, A>,
    pub allocated_bytes: usize,
    calls: HashMap<CallKey, HashSet<BufferKey>>,
    lanes: HashMap<LaneKey, Lane>,
    call_lanes: HashMap<RelayCall, usize>,
}

impl<B, I, A> TensorStore<B, I, A> {
    pub fn new(
        capacity: usize,
        byte_capacity: usize,
        request_capacity: usize,
        relay_depth: usize,
    ) -> Self {
        Self {
            capacity,
            byte_capacity,
            request_capacity,
            relay_depth,
            buffers: HashMap::new(),
            imports: HashMap::new(),
            arenas: HashMap::new(),
            views: HashMap::new(),
            allocated_bytes: 0,
            calls: HashMap::new(),
            lanes: HashMap::new(),
            call_lanes: HashMap::new(),
        }
    }
}

impl<B, I, A, T, C> TensorStore<B, I, A>
where
    B: Deref<Target = Mutex<Buffer<T, C>>>,
{
    /// Check the whole allocation group before the backend creates any views.
    pub fn validate_bindings(&self, bindings: &[(BufferId, String)], feature: bool) -> Result<()> {
        let mut seen = HashSet::new();
        let mut counts = HashMap::<String, usize>::new();
        for buffer in self.buffers.values() {
            let buffer = lock(buffer);
            if !buffer.feature && matches!(buffer.backing, Some(Backing::Persistent(_))) {
                *counts.entry(buffer.device.clone()).or_default() += 1;
            }
        }

        for (id, device) in bindings {
            let key = buffer_key(*id);
            if !seen.insert(key) || self.buffers.contains_key(&key) {
                return Err(Error::Invalid(
                    "persistent output is already registered".into(),
                ));
            }
            if !feature {
                let count = counts.entry(device.clone()).or_default();
                *count += 1;
                if *count > self.capacity {
                    return Err(Error::Resource(
                        "device-product arena has no query-ready free generation",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Capacity is committed only after numerical allocation succeeds.
    pub fn arena_capacity(&self, bytes: usize) -> Result<usize> {
        self.allocated_bytes
            .checked_add(bytes)
            .filter(|&projected| projected <= self.byte_capacity)
            .ok_or(Error::Resource("device-product byte capacity is exhausted"))
    }

    pub fn require_buffer(&self, buffer: &B) -> Result<()> {
        let value = lock(buffer);
        if value.backing.is_none()
            || !self
                .buffers
                .get(&buffer_key(value.id))
                .is_some_and(|current| std::ptr::eq(&**current, &**buffer))
        {
            return Err(Error::Invariant(
                "stale device-product physical generation".into(),
            ));
        }
        Ok(())
    }

    pub fn require_reference(&self, id: BufferId) -> Result<&B> {
        let buffer = self.buffers.get(&buffer_key(id))
            .filter(|buffer| lock(buffer).state == WriteState::Committed)
            .ok_or_else(|| Error::Invalid(format!(
                "unknown device-product reference: request {} produced by {:?} output {} generation {}",
                id.owner.request_id.0, id.producer_call_id, id.output_index, id.generation,
            )))?;
        if lock(buffer).id != id {
            return Err(Error::Invalid(
                "stale device-product logical generation".into(),
            ));
        }
        Ok(buffer)
    }

    pub fn commit_writes(&mut self, writes: &[B]) -> Result<()> {
        self.validate_writes(writes)?;
        let mut seen = HashSet::new();
        for write in writes {
            let value = lock(write);
            if value.state == WriteState::Deferred {
                continue;
            }
            if !seen.insert(buffer_key(value.id)) {
                return Err(Error::Invariant(
                    "device-product commit repeats an output".into(),
                ));
            }
        }

        for write in writes {
            let mut value = lock(write);
            if value.state == WriteState::Deferred {
                continue;
            }
            value.state = WriteState::Committed;
            self.calls
                .entry((value.id.owner, value.id.producer_call_id))
                .or_default()
                .insert(buffer_key(value.id));
        }
        Ok(())
    }

    pub fn validate_writes(&self, writes: &[B]) -> Result<()> {
        for write in writes {
            self.require_buffer(write)?;
            lock(write).require_candidate()?;
        }
        Ok(())
    }

    pub fn abandon_writes(&self, writes: &[B]) {
        for write in writes {
            if self.require_buffer(write).is_ok() {
                lock(write).released = true;
            }
        }
    }

    pub fn detach(&mut self, id: BufferId) {
        let call = (id.owner, id.producer_call_id);
        if let Some(buffers) = self.calls.get_mut(&call) {
            buffers.remove(&buffer_key(id));
            if buffers.is_empty() {
                self.calls.remove(&call);
            }
        }
    }

    pub fn release_calls(&mut self, calls: impl IntoIterator<Item = CallKey>) {
        for call in calls {
            if let Some(buffers) = self.calls.remove(&call) {
                for key in buffers {
                    if let Some(buffer) = self.buffers.get(&key) {
                        lock(buffer).released = true;
                    }
                }
            }
        }
    }

    pub fn release_buffers(&mut self, selected: &HashSet<BufferId>) -> Vec<BufferKey> {
        let keys = self
            .buffers
            .iter()
            .filter_map(|(&key, buffer)| selected.contains(&lock(buffer).id).then_some(key))
            .collect::<Vec<_>>();
        self.release_keys(&keys);
        keys
    }

    /// Retained features keep persistent storage across request completion;
    /// request-slot relay storage cannot outlive the slot's request.
    pub fn release_requests(
        &mut self,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Result<Vec<BufferKey>> {
        let mut keys = Vec::new();
        for (&key, buffer) in &self.buffers {
            let value = lock(buffer);
            if requests.contains(&value.id.owner) {
                if retained.contains(&value.id) {
                    if !matches!(value.backing, Some(Backing::Persistent(_))) {
                        return Err(Error::Invalid(
                            "finish cannot retain request-slot storage".into(),
                        ));
                    }
                } else {
                    keys.push(key);
                }
            }
        }
        self.release_keys(&keys);
        Ok(keys)
    }

    fn release_keys(&mut self, keys: &[BufferKey]) {
        for key in keys {
            if let Some(buffer) = self.buffers.get(key) {
                let id = lock(buffer).id;
                lock(buffer).released = true;
                self.detach(id);
            }
        }
    }

    /// Return retired owners and their backing only after the full query pass.
    /// The backend releases pool bindings and event references after this call.
    pub fn reclaim<TE, TC, CE, CC>(&mut self) -> Result<Vec<(B, Backing)>>
    where
        T: Deref<Target = Completion<TE, TC>>,
        C: Deref<Target = Completion<CE, CC>>,
    {
        let mut queried = HashMap::new();
        let mut ready = Vec::new();
        for (&key, buffer) in &self.buffers {
            if lock(buffer).access_retired(&mut queried)? {
                ready.push(key);
            }
        }

        let mut retired = Vec::with_capacity(ready.len());
        for key in ready {
            if let Some(buffer) = self.buffers.remove(&key) {
                let id = lock(&buffer).id;
                self.detach(id);
                let backing = self.release_backing(&buffer)?;
                retired.push((buffer, backing));
            }
        }
        Ok(retired)
    }

    pub fn release_backing(&mut self, buffer: &B) -> Result<Backing> {
        let backing = lock(buffer)
            .backing
            .take()
            .ok_or_else(|| Error::Invariant("tensor storage was retired more than once".into()))?;
        if let Backing::Relay((lane_key, dtype, field)) = &backing {
            let lane = self
                .lanes
                .get_mut(lane_key)
                .ok_or_else(|| Error::Invariant("request-relay lane lost its owner".into()))?;
            lane.fields.remove(&(dtype.clone(), *field));
            if lane.fields.is_empty() {
                self.call_lanes
                    .remove(&(lane_key.0.clone(), lane_key.1, lane.call.0, lane.call.1));
                self.lanes.remove(lane_key);
            }
        }
        Ok(backing)
    }

    pub fn relay_lane(&self, id: BufferId, device: &str, slot: usize) -> Result<usize> {
        let call = (device.to_owned(), slot, id.owner, id.producer_call_id);
        self.call_lanes
            .get(&call)
            .copied()
            .or_else(|| {
                (0..self.relay_depth)
                    .find(|&lane| !self.lanes.contains_key(&(device.to_owned(), slot, lane)))
            })
            .ok_or(Error::Resource(
                "request-relay unresolved window is exhausted",
            ))
    }

    pub fn bind_relay(&mut self, id: BufferId, relay: &RelayKey) -> Result<()> {
        let (lane_key, dtype, field) = relay;
        let lane = self.lanes.entry(lane_key.clone()).or_insert_with(|| Lane {
            call: (id.owner, id.producer_call_id),
            fields: HashSet::new(),
        });
        if !lane.fields.insert((dtype.clone(), *field)) {
            return Err(Error::Invariant(
                "request-relay lane was assigned more than once".into(),
            ));
        }
        self.call_lanes.insert(
            (
                lane_key.0.clone(),
                lane_key.1,
                id.owner,
                id.producer_call_id,
            ),
            lane_key.2,
        );
        Ok(())
    }

    /// After physical accesses drain, return owners for destruction outside
    /// the store lock. Existing numerical reads retain their tensor views.
    pub fn close(&mut self) -> Self {
        let empty = Self::new(
            self.capacity,
            self.byte_capacity,
            self.request_capacity,
            self.relay_depth,
        );
        std::mem::replace(self, empty)
    }
}

fn lock<T>(value: &Mutex<T>) -> MutexGuard<'_, T> {
    value.lock().unwrap_or_else(PoisonError::into_inner)
}
