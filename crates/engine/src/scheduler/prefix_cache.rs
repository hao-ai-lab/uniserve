//! Scheduler request-state adaptation for common KV prefix coordination.

use std::sync::atomic::Ordering;

use crate::kv::{BlockPool, KvCacheCoordinator, PrefixAdmission};

use crate::scheduler::{ReqState, SchedStats};

pub(crate) fn cached_blocks_for_admission(
    coordinator: &KvCacheCoordinator,
    state: &ReqState,
    pool: &BlockPool,
) -> PrefixAdmission {
    coordinator.cached_prefix_for_admission(
        pool,
        state.effective_prompt(),
        state.req.cache.read,
        !state.context.images.is_empty(),
        state.req.cache.isolation_key,
    )
}

pub(crate) fn lookup(
    coordinator: &KvCacheCoordinator,
    state: &mut ReqState,
    pool: &BlockPool,
    stats: &SchedStats,
) {
    let prompt = state.effective_prompt().to_vec();
    let lookup = coordinator
        .acquire_prefix(
            pool,
            &mut state.block_tables,
            &prompt,
            state.req.cache.read,
            !state.context.images.is_empty(),
            state.req.cache.isolation_key,
        )
        .expect("request cache groups match the KV coordinator");
    let block_size = pool.block_size();
    let full_blocks = prompt.len() / block_size;
    let query_blocks = if prompt.len().is_multiple_of(block_size) {
        full_blocks.saturating_sub(1)
    } else {
        full_blocks
    };
    stats
        .prefix
        .queries
        .fetch_add(query_blocks as u64, Ordering::Relaxed);
    stats
        .prefix
        .hits
        .fetch_add(lookup.cached_blocks as u64, Ordering::Relaxed);
    stats.prefix.hit_tokens.fetch_add(
        (lookup.cached_blocks * block_size) as u64,
        Ordering::Relaxed,
    );
    state.cursor.replay.block_hashes = lookup.block_hashes;
    state.cursor.replay.prefix_cached_blocks = lookup.cached_blocks;
    state.cursor.ingest.prompt_cursor = (lookup.cached_blocks * block_size) as u32;
    state.cursor.und.logical_pos = state.cursor.ingest.prompt_cursor;
    state.cursor.und.physical_kv_len = state.cursor.ingest.prompt_cursor;
}

pub(crate) fn cache_blocks(
    coordinator: &KvCacheCoordinator,
    state: &mut ReqState,
    pool: &BlockPool,
) {
    if state.cursor.replay.blocks_cached {
        return;
    }
    let prompt = state.effective_prompt().to_vec();
    if coordinator.cache_prefix(
        pool,
        &state.block_tables,
        &prompt,
        &state.cursor.replay.block_hashes,
        state.req.cache.write,
    ) {
        state.cursor.replay.blocks_cached = true;
    }
}
