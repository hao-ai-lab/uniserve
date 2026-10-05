//! Local double-buffered expert-weight reads from immutable peer storage.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, PoisonError};

use indexmap::IndexMap;

use crate::cuda::{CopyBatch, DeviceGuard, Event, Stream};
use crate::{Error, Result};

#[derive(Default)]
struct Call {
    started: bool,
    queued: Option<usize>,
    last: Option<usize>,
}

struct State<T> {
    backing: Option<T>,
    calls: Vec<Call>,
    closed: bool,
}

/// Prefetch the next expert layer into alternating remote-weight slots.
///
/// The numerical backend supplies immutable source mappings and destination
/// views. Calls serialize on their execution stream. External events protect
/// both copy-before-read and read-before-overwrite across CUDA graph segments.
pub struct WeightPrefetch<T> {
    device: i32,
    layers: HashMap<usize, usize>,
    copies: Vec<CopyBatch>,
    stream: Stream,
    ready: [Event; 2],
    consumed: [Event; 2],
    state: Mutex<State<T>>,
}

impl<T> WeightPrefetch<T> {
    /// Copy tuples are `(peer, destination, source, bytes)`, grouped by layer.
    /// `backing` retains every allocation used by those addresses.
    pub fn new(
        device: i32,
        layers: Vec<usize>,
        copies: Vec<Vec<(usize, usize, usize, usize)>>,
        backing: T,
    ) -> Result<Self> {
        if layers.len() != copies.len() || layers.is_empty() {
            return Err(Error::Invalid(
                "each expert layer needs one copy plan".into(),
            ));
        }

        let _device = DeviceGuard::new(device).map_err(Error::Cuda)?;
        Ok(Self {
            device,
            layers: layers
                .into_iter()
                .enumerate()
                .map(|(i, layer)| (layer, i))
                .collect(),
            copies: copies.into_iter().map(plan).collect(),
            stream: Stream::new().map_err(Error::Cuda)?,
            ready: std::array::from_fn(|_| Event::external(device)),
            consumed: std::array::from_fn(|_| Event::external(device)),
            state: Mutex::new(State {
                backing: Some(backing),
                calls: Vec::new(),
                closed: false,
            }),
        })
    }

    fn state(&self) -> MutexGuard<'_, State<T>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn contains(&self, module: usize) -> bool {
        self.layers.contains_key(&module)
    }

    fn index(&self, module: usize) -> Result<usize> {
        self.layers
            .get(&module)
            .copied()
            .ok_or_else(|| Error::Invalid("module has no prefetched expert weights".into()))
    }

    pub fn begin(&self) -> Result<()> {
        let mut state = self.state();
        if state.closed {
            return Err(Error::State("DWDP weight owner is closed"));
        }
        state.calls.push(Call::default());
        Ok(())
    }

    /// Join a speculative tail copy even if the call omitted its last layer.
    pub fn end(&self, compute: usize) -> Result<()> {
        let last = self
            .state()
            .calls
            .pop()
            .ok_or(Error::State("DWDP expert call requires an active context"))?
            .last;
        if let Some(last) = last {
            self.ready[last % 2].wait_on(compute).map_err(Error::Cuda)?;
        }
        Ok(())
    }

    /// `submit` either copies now or records a copy between graph segments.
    /// It runs outside the state lock and may call back into this owner.
    /// Callback errors retain their backend type in the inner result.
    pub fn before<E>(
        &self,
        module: usize,
        compute: usize,
        mut submit: impl FnMut(usize) -> std::result::Result<(), E>,
    ) -> Result<std::result::Result<(), E>> {
        let index = self.index(module)?;
        let (started, queued) = {
            let mut state = self.state();
            let call = state
                .calls
                .last_mut()
                .ok_or(Error::State("DWDP expert call requires an active context"))?;
            let started = std::mem::replace(&mut call.started, true);
            (started, call.queued)
        };

        let _device = DeviceGuard::new(self.device).map_err(Error::Cuda)?;
        if !started {
            for event in &self.consumed {
                event.record(compute).map_err(Error::Cuda)?;
            }
        }

        if queued != Some(index)
            && let Err(error) = self.prefetch(index, &mut submit)?
        {
            return Ok(Err(error));
        }

        self.ready[index % 2]
            .wait_on(compute)
            .map_err(Error::Cuda)?;

        let next = (index + 1 < self.copies.len()).then_some(index + 1);
        self.state()
            .calls
            .last_mut()
            .ok_or(Error::State("DWDP expert call requires an active context"))?
            .queued = next;
        if let Some(next) = next {
            return self.prefetch(next, &mut submit);
        }
        Ok(Ok(()))
    }

    fn prefetch<E>(
        &self,
        index: usize,
        submit: &mut impl FnMut(usize) -> std::result::Result<(), E>,
    ) -> Result<std::result::Result<(), E>> {
        let mut state = self.state();
        let call = state
            .calls
            .last_mut()
            .ok_or(Error::State("DWDP expert call requires an active context"))?;
        call.last = Some(index);
        drop(state);
        Ok(submit(index))
    }

    pub fn after(&self, module: usize, compute: usize) -> Result<()> {
        let index = self.index(module)?;
        self.consumed[index % 2]
            .record(compute)
            .map_err(Error::Cuda)
    }

    /// Replay a recorded peer read, preserving the same two-slot dependencies
    /// as eager execution. No host synchronization is performed.
    pub fn copy(&self, index: usize) -> Result<()> {
        if self.state().closed {
            return Err(Error::State("DWDP weight owner is closed"));
        }

        let copy = self
            .copies
            .get(index)
            .ok_or_else(|| Error::Invalid("expert layer is outside the copy plan".into()))?;
        let _device = DeviceGuard::new(self.device).map_err(Error::Cuda)?;
        let stream = self.stream.handle();
        self.consumed[index % 2]
            .wait_on(stream)
            .map_err(Error::Cuda)?;
        copy.copy_on(stream).map_err(Error::Cuda)?;
        self.ready[index % 2].record(stream).map_err(Error::Cuda)
    }

    /// The caller retires numerical readers first. Drain the copy stream and
    /// return backing for destruction outside owner locks and binding borrows.
    pub fn close(&self) -> Result<Option<T>> {
        let mut state = self.state();
        if state.backing.is_none() {
            return Ok(None);
        }
        if !state.calls.is_empty() {
            return Err(Error::State("cannot close active weight prefetch"));
        }
        state.closed = true;
        drop(state);

        self.stream.wait().map_err(Error::Cuda)?;
        Ok(self.state().backing.take())
    }

    pub fn visit<R>(&self, visit: impl FnOnce(&T) -> R) -> Option<R> {
        self.state().backing.as_ref().map(visit)
    }
}

impl<T> Drop for WeightPrefetch<T> {
    #[expect(
        clippy::mem_forget,
        reason = "unknown copy completion must retain peer mappings"
    )]
    fn drop(&mut self) {
        if self.close().is_err() {
            let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
            std::mem::forget(state.backing.take());
        }
    }
}

fn plan(copies: Vec<(usize, usize, usize, usize)>) -> CopyBatch {
    // As in TensorRT-LLM's batched DWDP reads, interleave 2 MiB slices from
    // different peers. Destinations are disjoint, allowing DMA pipelining.
    let mut peers = IndexMap::<usize, Vec<(usize, usize, usize)>>::new();
    const CHUNK: usize = 2 << 20;

    for (peer, destination, source, bytes) in copies {
        let slices = peers.entry(peer).or_default();
        for offset in (0..bytes).step_by(CHUNK) {
            slices.push((
                destination + offset,
                source + offset,
                CHUNK.min(bytes - offset),
            ));
        }
    }

    let rounds = peers.values().map(Vec::len).max().unwrap_or(0);
    CopyBatch::new((0..rounds).flat_map(|round| {
        peers
            .values()
            .filter_map(move |peer| peer.get(round).copied())
    }))
}
