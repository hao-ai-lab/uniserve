//! Automatic prefix-cache coordination: per-block prompt hashing, the
//! admission-time longest-cached-prefix lookup, reuse on admission, post-prefill
//! publication, and prefix-hit accounting. The KV block pool ([`BlockManager`])
//! and per-request state ([`ReqState`]) stay owned by the coordinator and are
//! threaded in as borrows.

use std::sync::atomic::Ordering;

use uniserve_core::HashAlgo;
use uniserve_kv::{BlockManager, BlockState, block_hash, modality_tag};

use crate::scheduler::{ReqState, SchedStats};

/// Owns the prefix-cache toggle and hash configuration. Hashing is content-keyed
/// per full prompt block; image-bearing prompts and opt-out requests are excluded.
pub(crate) struct PrefixCacheCoordinator {
    enable: bool,
    hash_algo: HashAlgo,
    hash_seed: u64,
}

impl PrefixCacheCoordinator {
    pub(crate) fn new() -> Self {
        Self {
            enable: true,
            hash_algo: HashAlgo::Fnv1a,
            hash_seed: 0,
        }
    }

    pub(crate) fn set_enabled(&mut self, on: bool) {
        self.enable = on;
    }

    pub(crate) fn set_hash_algo(&mut self, algo: HashAlgo) {
        self.hash_algo = algo;
    }

    /// on admission, find the longest cached prompt-block prefix that exists,
    /// returning `(cached, cached_free)` — total reusable blocks and how many of
    /// those are currently in the free/cached state (not pinned by a running
    /// request). Read-only; the actual reuse happens in [`Self::lookup`].
    pub(crate) fn cached_blocks_for_admission(
        &self,
        st: &ReqState,
        bm: &BlockManager,
        block_size: usize,
    ) -> (usize, usize) {
        if !self.enable || !st.req.cache.read || !st.context.images.is_empty() {
            return (0, 0);
        }
        let prompt = st.effective_prompt();
        let bs = block_size;
        if bs == 0 || prompt.len() < bs {
            return (0, 0);
        }
        let num_full = prompt.len() / bs;
        let lookup_limit = if prompt.len().is_multiple_of(bs) {
            num_full.saturating_sub(1)
        } else {
            num_full
        };
        let mut cached = 0usize;
        let mut cached_free = 0usize;
        let mut parent = self.request_hash_seed(st);
        for i in 0..lookup_limit {
            let toks = &prompt[i * bs..(i + 1) * bs];
            let h = block_hash(
                parent,
                0,
                modality_tag(uniserve_core::Modality::Und),
                toks,
                self.hash_algo,
            );
            // verify the candidate tokens against the cached block.
            match bm.lookup_cached(h, toks) {
                Some(b) => {
                    if bm.block_state(b) == BlockState::Cached {
                        cached_free += 1;
                    }
                }
                None => break,
            }
            cached += 1;
            parent = h;
        }
        (cached, cached_free)
    }

    /// on admission, find the longest cached prompt-block prefix, reuse those
    /// blocks (bump ref_cnt) and advance `prompt_cursor` past them so prefill
    /// only computes the uncached suffix. This is `get_computed_blocks` in Rust.
    /// Only text (`Und`) prefixes are cached; image latents are excluded
    /// (modality-tagged hashes).
    pub(crate) fn lookup(
        &self,
        st: &mut ReqState,
        bm: &mut BlockManager,
        stats: &SchedStats,
        block_size: usize,
    ) {
        if !self.enable {
            return;
        }
        let id = st.req.request_id;
        // Prompts with spliced image embeddings use the encoder cache rather
        // than text-prefix blocks.
        if !st.context.images.is_empty() {
            return;
        }
        let prompt = st.effective_prompt().to_vec();
        let bs = block_size;
        if bs == 0 || prompt.len() < bs {
            return;
        }
        let num_full = prompt.len() / bs;
        // Always leave at least the final token to (re)compute, so a fully
        // block-aligned prompt still produces logits: cap the lookup at the
        // number of *complete* blocks that precede the last token.
        let lookup_limit = if prompt.len().is_multiple_of(bs) {
            num_full.saturating_sub(1)
        } else {
            num_full
        };

        let mut hashes = Vec::with_capacity(num_full);
        let mut parent = self.request_hash_seed(st);
        for i in 0..num_full {
            let toks = &prompt[i * bs..(i + 1) * bs];
            let h = block_hash(
                parent,
                0,
                modality_tag(uniserve_core::Modality::Und),
                toks,
                self.hash_algo,
            );
            hashes.push(h);
            parent = h;
        }

        st.replay.block_hashes = hashes.clone();
        if !st.req.cache.read {
            return;
        }

        let mut cached = 0usize;
        for (i, &h) in hashes.iter().take(lookup_limit).enumerate() {
            // pass the block's own tokens so the cache verifies content,
            // not just the 64-bit hash, before reusing the block.
            let toks = &prompt[i * bs..(i + 1) * bs];
            if let Some(b) = bm.lookup_cached(h, toks)
                && bm.acquire_cached(id, b, h, toks)
            {
                cached += 1;
                continue;
            }
            break;
        }

        stats
            .prefix
            .queries
            .fetch_add(lookup_limit as u64, Ordering::Relaxed);
        stats
            .prefix
            .hits
            .fetch_add(cached as u64, Ordering::Relaxed);
        stats
            .prefix
            .hit_tokens
            .fetch_add((cached * bs) as u64, Ordering::Relaxed);

        st.replay.prefix_cached_blocks = cached;
        st.ingest.prompt_cursor = (cached * bs) as u32;
        st.und.logical_pos = st.ingest.prompt_cursor;
        // Reused blocks are physically resident in the request's block table:
        // the next physical write lands directly after them, and every
        // capacity computation derived from the physical cursor must cover
        // the reused span.
        st.und.physical_kv_len = st.ingest.prompt_cursor;
    }

    /// after a request's prompt is fully prefilled, publish its full prompt
    /// blocks to the prefix cache so later requests can reuse them. Idempotent
    /// (shared/reused blocks are already mapped).
    pub(crate) fn cache_blocks(&self, st: &mut ReqState, bm: &mut BlockManager, block_size: usize) {
        if !self.enable || !st.req.cache.write {
            return;
        }
        if st.replay.blocks_cached {
            return;
        }
        let id = st.req.request_id;
        let hashes = st.replay.block_hashes.clone();
        let prompt = st.effective_prompt().to_vec();
        let bs = block_size;
        let blocks = bm.blocks_for_group(id, 0).to_vec();
        for (i, h) in hashes.iter().enumerate() {
            // store the block's own tokens with its hash so later hits can
            // verify content. Block i was hashed from prompt[i*bs..(i+1)*bs] (the
            // same slice prefix_lookup used to compute `block_hashes`).
            let (start, end) = (i * bs, (i + 1) * bs);
            if let (Some(b), Some(toks)) = (blocks.get(i), prompt.get(start..end)) {
                bm.cache_block(*b, *h, toks);
            }
        }
        st.replay.blocks_cached = true;
    }

    fn request_hash_seed(&self, st: &ReqState) -> u64 {
        st.req
            .cache
            .isolation_key
            .map_or(self.hash_seed, |key| self.hash_seed ^ key.rotate_left(17))
    }
}
