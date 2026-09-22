//! Request backing and retained encoder publications owned by the scheduler.

use super::*;
use uniserve_worker_ipc::BufferId;

impl Scheduler {
    pub(super) fn free_media_request(&mut self, worker: &crate::WorkerId, allocation: RequestSlot) {
        self.media_memory
            .get_mut(worker)
            .expect("media allocation names a loaded worker")
            .requests
            .free(allocation);
    }

    pub(super) fn free_media_buffer(&mut self, worker: &crate::WorkerId, allocation: BufferSpan) {
        self.media_memory
            .get_mut(worker)
            .expect("media allocation names a loaded worker")
            .buffers
            .free(allocation);
    }

    pub(super) fn cache(&self) -> &KVCacheManager {
        self.cache
            .as_ref()
            .expect("generation requires loaded KV cache")
    }

    pub(super) fn free_blocks(&self) -> usize {
        self.cache
            .as_ref()
            .map_or(0, |cache| cache.block_pool.free_request_pages())
    }

    pub(super) fn usable_blocks(&self) -> usize {
        self.cache.as_ref().map_or(0, |cache| cache.usable_blocks)
    }

    /// Transfer the existing Tensor allocation to encoder-cache retention.
    pub(super) fn retain_encoder_buffer(
        &mut self,
        buffer: BufferId,
        allocation: BufferSpan,
    ) -> Result<(), BufferSpan> {
        if allocation.owner != buffer.owner || self.encoder_buffers.contains_key(&buffer) {
            return Err(allocation);
        }
        self.encoder_buffers.insert(buffer, allocation);
        Ok(())
    }

    pub(super) fn take_encoder_buffer(&mut self, buffer: BufferId) -> Option<BufferSpan> {
        self.encoder_buffers.remove(&buffer)
    }

    /// Buffers whose cache/reader lifetime extends beyond their producing request.
    pub(super) fn retained_buffers(&self, request: RequestKey) -> Vec<BufferId> {
        let mut buffers: Vec<_> = self
            .encoder_buffers
            .keys()
            .copied()
            .filter(|buffer| buffer.owner == request)
            .collect();
        buffers.sort_unstable_by_key(|buffer| {
            (
                buffer.producer_call_id,
                buffer.output_index,
                buffer.generation,
            )
        });
        buffers
    }
}
