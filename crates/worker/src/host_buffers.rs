//! Host input buffers retained until their asynchronous copies complete.

use crate::cuda::{DeviceGuard, Event};
use crate::{Error, Result};

/// A fixed ring of host copy sources.
///
/// The numerical backend allocates and fills the buffers. After enqueueing a
/// copy, it records that slot on the copy stream before acquiring it again.
/// Acquiring a slot waits only for its previous copy; unrelated device work
/// does not prevent reuse. Slots are round-robin, not individually leased.
pub struct HostBuffers<T> {
    buffers: Vec<T>,
    events: Vec<Event>,
    device: Option<i32>,
    cursor: usize,
}

impl<T> HostBuffers<T> {
    pub fn new(buffers: Vec<T>, device: Option<i32>) -> Result<Self> {
        if buffers.is_empty() {
            return Err(Error::Invalid("host buffer depth must be positive".into()));
        }

        let events = match device {
            Some(device) => buffers
                .iter()
                .map(|_| Event::new(device, false, false))
                .collect(),
            None => Vec::new(),
        };

        Ok(Self {
            buffers,
            events,
            device,
            cursor: 0,
        })
    }

    pub fn device(&self) -> Option<i32> {
        self.device
    }

    pub fn buffers(&self) -> &[T] {
        &self.buffers
    }

    /// Wait for the next slot's last copy before returning its source buffer.
    pub fn acquire(&mut self) -> Result<(usize, &T)> {
        if self.buffers.is_empty() {
            return Err(Error::State("host buffers are closed"));
        }

        let slot = self.cursor;
        if let Some(event) = self.events.get(slot)
            && !event.ready().map_err(Error::Cuda)?
        {
            event.wait().map_err(Error::Cuda)?;
        }

        self.cursor = (slot + 1) % self.buffers.len();
        Ok((slot, &self.buffers[slot]))
    }

    /// Fence the copy just submitted from `slot` on the supplied CUDA stream.
    /// Non-CUDA buffers do not need copy fences.
    #[expect(
        clippy::mem_forget,
        reason = "failed copy fencing cannot release its source allocations"
    )]
    pub fn record_copy(&mut self, slot: usize, stream: usize) -> Result<()> {
        let Some(device) = self.device else {
            return Ok(());
        };
        let event = self
            .events
            .get(slot)
            .ok_or_else(|| Error::Invalid("host buffer slot is outside its ring".into()))?;

        let result = DeviceGuard::new(device).and_then(|_device| event.record(stream));
        if result.is_err() {
            // The copy was submitted before this fence. If recording fails,
            // its sources must neither be reused nor released as completed.
            std::mem::forget(std::mem::take(&mut self.buffers));
            self.events.clear();
        }
        result.map_err(Error::Cuda)
    }

    /// Drain copies and return their source allocations for release.
    ///
    /// Bindings drop these allocations outside their owner lock, since a
    /// foreign tensor destructor can call back into the runtime.
    pub fn close(&mut self) -> Result<Vec<T>> {
        for event in &self.events {
            if !event.ready().map_err(Error::Cuda)? {
                event.wait().map_err(Error::Cuda)?;
            }
        }

        self.events.clear();
        Ok(std::mem::take(&mut self.buffers))
    }
}

impl<T> Drop for HostBuffers<T> {
    #[expect(
        clippy::mem_forget,
        reason = "unknown device completion must retain its source allocations"
    )]
    fn drop(&mut self) {
        if self.close().is_err() {
            // A failed CUDA wait cannot establish that the source is reusable.
            // Keep its allocation alive rather than free memory still in use.
            std::mem::forget(std::mem::take(&mut self.buffers));
        }
    }
}
