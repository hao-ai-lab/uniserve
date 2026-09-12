//! Request backing and retained encoder publications owned by the scheduler.

use super::*;
use uniserve_worker_ipc::BufferId;

impl Scheduler {
    pub(super) fn free_allocation(&mut self, allocation: Allocation) {
        match allocation {
            Allocation::RequestSlot { .. } => self.request_pool.free(allocation),
            Allocation::Kv { .. } => drop(allocation),
            Allocation::Latent { .. } => self.latent_pool.free(allocation),
            Allocation::Buffer { .. } => self.buffer_pool.free(allocation),
        }
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
        allocation: Allocation,
    ) -> Result<(), Allocation> {
        if allocation.owner() != buffer.owner
            || !matches!(allocation, Allocation::Buffer { .. })
            || self.encoder_buffers.contains_key(&buffer)
        {
            return Err(allocation);
        }
        self.encoder_buffers.insert(buffer, allocation);
        Ok(())
    }

    pub(super) fn take_encoder_buffer(&mut self, buffer: BufferId) -> Option<Allocation> {
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
                buffer.producer_op_id,
                buffer.output_index,
                buffer.generation,
            )
        });
        buffers
    }
}
