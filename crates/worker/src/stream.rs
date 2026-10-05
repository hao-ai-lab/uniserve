//! Execution streams and reusable ingress/output fences.

use std::sync::Arc;

use crate::cuda::{DeviceGuard, Event, Stream};

/// An execution stream, its SM allocation and bounded submission fences.
///
/// Callers serialize context use across streams sharing an SM partition.
/// Output fences may be borrowed until their ring slot is reused; forks
/// independently retain an owned SM partition.
pub struct CUDAStream {
    device: i32,
    stream: Option<Stream>,
    sm_count: u32,
    partitioned: bool,
    ingress: Vec<Arc<Event>>,
    outputs: Vec<Arc<Event>>,
    cursor: usize,
}

impl CUDAStream {
    fn new(device: i32, stream: Stream, event_slots: usize) -> Result<Self, String> {
        if event_slots == 0 {
            return Err("CUDA stream requires a positive event bound".into());
        }

        let sm_count = stream.sm_count()?;
        let partitioned = stream.partitioned()?;
        let outputs = (0..event_slots)
            .map(|_| Arc::new(stream.event(device)))
            .collect();

        Ok(Self {
            device,
            stream: Some(stream),
            sm_count,
            partitioned,
            ingress: (0..event_slots)
                .map(|_| Arc::new(Event::new(device, false, false)))
                .collect(),
            outputs,
            cursor: 0,
        })
    }

    /// Borrow an existing numerical stream; its creator retains the handle.
    pub fn borrowed(device: i32, handle: usize, event_slots: usize) -> Result<Self, String> {
        let _device = DeviceGuard::new(device)?;
        Self::new(device, Stream::borrowed(handle), event_slots)
    }

    /// Own an independent stream in the origin's borrowed CUDA context.
    pub fn sibling(device: i32, origin: usize, event_slots: usize) -> Result<Self, String> {
        let _device = DeviceGuard::new(device)?;
        Self::new(device, Stream::sibling(origin)?, event_slots)
    }

    /// Allocate the configured SM partitions from one disjoint split tree.
    pub fn partition(device: i32, counts: &[u32], slots: &[usize]) -> Result<Vec<Self>, String> {
        if counts.len() != slots.len() || slots.contains(&0) {
            return Err("each CUDA stream requires a positive event-slot count".into());
        }

        Stream::partition(device, counts)?
            .into_iter()
            .zip(slots)
            .map(|(stream, slots)| Self::new(device, stream, *slots))
            .collect()
    }

    pub fn fork(&self) -> Result<Self, String> {
        let _device = DeviceGuard::new(self.device)?;
        Self::new(self.device, self.stream()?.fork()?, self.ingress.len())
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub fn sm_count(&self) -> u32 {
        self.sm_count
    }

    pub fn full_device(&self) -> bool {
        !self.partitioned
    }

    pub fn closed(&self) -> bool {
        self.stream.is_none()
    }

    pub fn handle(&self) -> Result<usize, String> {
        Ok(self.stream()?.handle())
    }

    fn stream(&self) -> Result<&Stream, String> {
        self.stream
            .as_ref()
            .ok_or_else(|| "CUDA stream is closed".into())
    }

    /// Enqueue a device dependency without waiting for the producer on the host.
    pub fn wait(&self, producer: usize) -> Result<(), String> {
        let stream = self.handle()?;
        if stream == producer {
            return Ok(());
        }

        let _device = DeviceGuard::new(self.device)?;
        let event = &self.ingress[self.cursor];
        event.record(producer)?;
        event.wait_on(stream)
    }

    /// Fence one submission for later joins, preserving independent lane overlap.
    /// Consumers enqueue their waits before this ring slot is reused.
    pub fn record(&mut self, consumer: usize) -> Result<Option<Arc<Event>>, String> {
        let stream = self.handle()?;
        if stream == consumer {
            return Ok(None);
        }

        let _device = DeviceGuard::new(self.device)?;
        let event = &self.outputs[self.cursor];
        event.record(stream)?;
        self.cursor = (self.cursor + 1) % self.outputs.len();
        Ok(Some(Arc::clone(event)))
    }

    pub fn synchronize(&self) -> Result<(), String> {
        let _device = DeviceGuard::new(self.device)?;
        self.stream()?.wait()
    }

    /// Drain submitted work before releasing the stream and its SM partition.
    /// Failed collective execution retains resources until process exit.
    #[allow(clippy::mem_forget)]
    pub fn close(&mut self, aborted: bool) -> Result<(), String> {
        if aborted {
            std::mem::forget(self.stream.take());
            std::mem::forget(std::mem::take(&mut self.ingress));
            std::mem::forget(std::mem::take(&mut self.outputs));
            return Ok(());
        }

        if let Some(stream) = &self.stream {
            let _device = DeviceGuard::new(self.device)?;
            stream.wait()?;
        }

        self.ingress.clear();
        self.outputs.clear();
        self.stream = None;
        Ok(())
    }
}

impl Drop for CUDAStream {
    fn drop(&mut self) {
        if self.close(false).is_err() {
            let _ = self.close(true);
        }
    }
}
